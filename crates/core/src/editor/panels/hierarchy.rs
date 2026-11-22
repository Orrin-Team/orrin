//! The scene as a tree: every entity, its kind and its children, with search,
//! drag to reparent, and the scene's counts at the foot. A prefab dragged in
//! from the asset browser is placed under whatever row it is let go over, and
//! the rows of a prefab instance wear the prefab colour, with what can be done
//! to the instance on the right-click menu. A folder is a row with no place of
//! its own (`scene::Folder`), and a row's eye hides it from the editor's view
//! and nowhere else (`scene::editor_hidden`).
//!
//! The row menu and the Edit menu duplicate, copy and paste the selection's
//! trees (`scene::prefab::copy`), and Rename's key (F2 unless rebound)
//! renames the primary in its row. While the pointer is over the tree, or a
//! press last landed in it, the arrows walk the rows; they are the tree's own
//! and not commands, so they are not rebound. A selection made anywhere else — in the scene, by an
//! undo, by a command — opens the rows above it and scrolls to it; one made
//! by a click here leaves the list where it is, since moving it would move
//! what was aimed at. A row cannot be dropped between two others: siblings
//! have no order of their own to change, only their slots in the ECS, and a
//! scene file writes them sorted by id.
//!
//! Rows are laid out flat, one fixed-height rect each, rather than as nested
//! collapsing headers. That is what lets a row fill the panel edge to edge —
//! selection and hover both cover the whole line, indent included — and what
//! keeps the pointer from moving anything: the hover affordances are painted
//! over a row that is already the size it will ever be. It is also what lets
//! the list draw only the rows on screen, which a tree of nested `Ui`s cannot.

use std::collections::HashSet;

use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, EntityId};

use crate::editor::commands::{self, Command};
use crate::editor::kind::Kind;
use crate::editor::state::{EditorState, PrefabAction};
use crate::editor::{icons, theme, widgets};
use crate::scene::{Hierarchy, Name, PrefabInstance, Prefabs};

/// One row of the tree, resolved before any drawing starts.
///
/// The panel reads the world to build these and then never touches it again,
/// which is what lets a drag reshape the hierarchy without the draw walking a
/// tree that is changing underneath it.
struct Node {
    entity: Entity,
    /// What the row's open state is kept under across sessions; see
    /// `state::ClosedRows`.
    id: Option<EntityId>,
    label: String,
    /// Whether `label` is a [`Name`] rather than a stand-in for one, and so
    /// can be renamed in the row.
    named: bool,
    kind: Kind,
    children: Vec<usize>,
    is_root: bool,
    /// Whether the search query keeps this row. Filtering happens here rather
    /// than in `snapshot` because a node whose *child* matches has to stay
    /// visible — drop it and the tree collapses out from under the match.
    visible: bool,
    prefab: Role,
    /// Whether it can be picked in the scene: [`Locked`](crate::scene::Locked).
    lock: Mark,
    /// Whether the editor's view leaves it out: see `scene::editor_hidden`.
    hidden: Mark,
    folder: Option<crate::scene::Folder>,
}

/// A row's state for a toggle whose effect reaches everything under it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mark {
    Off,
    /// Set on the row itself.
    Own,
    /// Under a row that has it set.
    Inherited,
}

/// Where a row stands in a prefab instance.
#[derive(Clone, Debug, PartialEq)]
enum Role {
    None,
    /// The instance root, with its prefab's name for the tooltip.
    Root(String),
    Member,
}

/// A node as it lands on screen: which one, how deep, whether it has a
/// disclosure to draw, and whether that is open.
struct Row {
    node: usize,
    depth: u32,
    parent: bool,
    open: bool,
}

const ROW: f32 = 24.0;
const INDENT: f32 = 14.0;
pub(super) const FOOTER: f32 = 24.0;

/// What the tree keeps between frames, in egui's memory: the panel is drawn
/// afresh each frame and the rest of its state is the scene's.
#[derive(Clone, Default)]
struct Tree {
    /// The primary selection as the tree last left it. One that differs at
    /// the start of a frame was chosen somewhere else, and is revealed.
    seen: Option<Entity>,
    /// Whether the last press landed in the tree. The arrows are its while
    /// the pointer is over it or until a press lands elsewhere.
    clicked: bool,
    renaming: Option<Renaming>,
}

/// A name being edited in its row. Enter renames; Escape, or clicking away,
/// does not.
#[derive(Clone)]
struct Renaming {
    entity: Entity,
    text: String,
    started: bool,
    /// Whether the row was drawn this frame. One scrolled away cannot be
    /// typed into, so its edit is dropped rather than left waiting.
    drawn: bool,
}

fn tree_id() -> egui::Id {
    egui::Id::new("hierarchy_tree")
}

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    let ctx = ui.ctx().clone();
    let panel = ui.max_rect();
    let mut tree: Tree = ctx
        .data(|data| data.get_temp(tree_id()))
        .unwrap_or_default();
    let press = ui.input(|input| {
        input
            .pointer
            .any_pressed()
            .then(|| input.pointer.interact_pos())
            .flatten()
    });
    if let Some(at) = press {
        tree.clicked = panel.contains(at);
    }
    let active = (ui.rect_contains_pointer(panel) || tree.clicked) && !ctx.wants_keyboard_input();

    let primary = state.selection.primary().filter(|&e| world.is_alive(e));
    let reveal = primary.filter(|&primary| Some(primary) != tree.seen);
    if let Some(entity) = reveal {
        open_ancestors(&ctx, world, state, entity);
    }
    let nodes = snapshot(world, &state.hierarchy_query);
    let searching = !state.hierarchy_query.trim().is_empty();

    let rename = state.take_rename_request()
        || (active && ui.input(|input| state.settings.keys.pressed(input, Command::Rename)));
    let mut scroll_to = reveal;
    if rename
        && let Some(node) = primary.and_then(|p| nodes.iter().find(|n| n.entity == p && n.named))
    {
        tree.renaming = Some(Renaming {
            entity: node.entity,
            text: node.label.clone(),
            started: false,
            drawn: false,
        });
        scroll_to = Some(node.entity);
    }

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

    let mut rows = flatten(&ctx, &nodes, searching, state.closed_rows());
    if active
        && tree.renaming.is_none()
        && let Some(moved) = navigate(&ctx, &nodes, &rows, searching, state)
    {
        // A row opened or closed by the keys reshapes the list.
        rows = flatten(&ctx, &nodes, searching, state.closed_rows());
        scroll_to = Some(moved);
    }
    let scroll_row =
        scroll_to.and_then(|entity| rows.iter().position(|row| nodes[row.node].entity == entity));
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
        let mut scroll = egui::ScrollArea::vertical()
            .id_salt("hierarchy_rows")
            .auto_shrink(false);
        // The least scroll that brings the row fully into view. Only for a
        // row chosen by the keys or from outside the tree: one clicked is
        // under the pointer already, and moving it would move what was aimed
        // at.
        if let Some(index) = scroll_row {
            let id = ui.make_persistent_id(egui::Id::new("hierarchy_rows"));
            let offset =
                egui::scroll_area::State::load(&ctx, id).map_or(0.0, |scrolled| scrolled.offset.y);
            let (top, bottom) = (index as f32 * ROW, (index + 1) as f32 * ROW);
            let height = ui.available_height();
            let wanted = if top < offset {
                top
            } else if bottom > offset + height {
                bottom - height
            } else {
                offset
            };
            scroll = scroll.vertical_scroll_offset(wanted);
        }
        scroll.show_rows(ui, ROW, rows.len(), |ui, range| {
            for row in &rows[range] {
                draw_row(ui, &nodes, &rows, row, state, &mut tree.renaming);
            }
        });
    });
    if let Some(renaming) = &mut tree.renaming {
        if renaming.started && !renaming.drawn {
            tree.renaming = None;
        } else {
            renaming.drawn = false;
        }
    }
    tree.seen = state.selection.primary();
    ctx.data_mut(|data| data.insert_temp(tree_id(), tree));
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

/// A row with an id is open unless the scene's closed rows name it, which
/// outlive the session; one without is kept in egui's memory, for this
/// session only.
fn is_open(ctx: &egui::Context, node: &Node, closed: Option<&HashSet<EntityId>>) -> bool {
    match node.id {
        Some(id) => closed.is_none_or(|closed| !closed.contains(&id)),
        None => ctx
            .data(|data| data.get_temp::<bool>(open_id(node.entity)))
            .unwrap_or(true),
    }
}

/// The rows on screen, in order. A search opens every branch on the way to a
/// match, so a match is never hidden inside a closed parent.
fn flatten(
    ctx: &egui::Context,
    nodes: &[Node],
    searching: bool,
    closed: Option<&HashSet<EntityId>>,
) -> Vec<Row> {
    fn walk(
        ctx: &egui::Context,
        nodes: &[Node],
        index: usize,
        depth: u32,
        searching: bool,
        closed: Option<&HashSet<EntityId>>,
        out: &mut Vec<Row>,
    ) {
        let node = &nodes[index];
        let parent = node.children.iter().any(|&child| nodes[child].visible);
        let open = is_open(ctx, node, closed);
        out.push(Row {
            node: index,
            depth,
            parent,
            open,
        });
        if parent && (searching || open) {
            for &child in &node.children {
                if nodes[child].visible {
                    walk(ctx, nodes, child, depth + 1, searching, closed, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for index in 0..nodes.len() {
        if nodes[index].is_root && nodes[index].visible {
            walk(ctx, nodes, index, 0, searching, closed, &mut out);
        }
    }
    out
}

fn set_open(ctx: &egui::Context, state: &mut EditorState, node: &Node, open: bool) {
    match node.id {
        Some(id) => state.set_row_open(id, open),
        None => ctx.data_mut(|data| data.insert_temp(open_id(node.entity), open)),
    }
}

/// Open every row above `entity`, so its own is in the list.
fn open_ancestors(ctx: &egui::Context, world: &World, state: &mut EditorState, entity: Entity) {
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        let Some(parent) = crate::scene::persist::live_parent(world, cursor) else {
            return;
        };
        match world.get::<EntityId>(parent).map(|id| *id) {
            Some(id) => state.set_row_open(id, true),
            None => ctx.data_mut(|data| data.insert_temp(open_id(parent), true)),
        }
        cursor = parent;
    }
}

/// The arrows over the rows on screen: Up and Down move the selection, with
/// Shift extending it; Left closes a row, or climbs to its parent; Right
/// opens one, or descends to its first child. Returns the row the keys
/// landed on, for the list to keep in view.
fn navigate(
    ctx: &egui::Context,
    nodes: &[Node],
    rows: &[Row],
    searching: bool,
    state: &mut EditorState,
) -> Option<Entity> {
    use egui::{Key, Modifiers};
    // Shift first: a pattern without it matches a press with it.
    let arrow = |input: &mut egui::InputState, key| {
        if input.consume_key(Modifiers::SHIFT, key) {
            Some(true)
        } else {
            input.consume_key(Modifiers::NONE, key).then_some(false)
        }
    };
    let (up, down, left, right) = ctx.input_mut(|input| {
        (
            arrow(input, Key::ArrowUp),
            arrow(input, Key::ArrowDown),
            input.consume_key(Modifiers::NONE, Key::ArrowLeft),
            input.consume_key(Modifiers::NONE, Key::ArrowRight),
        )
    });
    if rows.is_empty() {
        return None;
    }
    let order: Vec<Entity> = rows.iter().map(|row| nodes[row.node].entity).collect();
    let at = state
        .selection
        .primary()
        .and_then(|primary| order.iter().position(|&entity| entity == primary));
    if let Some(extend) = down.or(up) {
        let to = match at {
            None => 0,
            Some(at) if down.is_some() => (at + 1).min(order.len() - 1),
            Some(at) => at.saturating_sub(1),
        };
        if extend {
            state.selection.extend_to(order[to], &order);
        } else {
            state.selection.set(order[to]);
        }
        return Some(order[to]);
    }
    let at = at?;
    let row = &rows[at];
    let node = &nodes[row.node];
    // A search shows every branch open whatever is kept, so the keys only
    // move through it.
    let open = row.parent && (searching || row.open);
    if left {
        if open && !searching {
            set_open(ctx, state, node, false);
            return Some(node.entity);
        }
        let parent = rows[..at]
            .iter()
            .rposition(|above| above.depth < row.depth)?;
        state.selection.set(order[parent]);
        return Some(order[parent]);
    }
    if right {
        if row.parent && !open {
            set_open(ctx, state, node, true);
            return Some(node.entity);
        }
        if open
            && rows
                .get(at + 1)
                .is_some_and(|below| below.depth > row.depth)
        {
            state.selection.set(order[at + 1]);
            return Some(order[at + 1]);
        }
    }
    None
}

/// Select every row the tree lists: what the filter keeps, inside rows that
/// are open. The primary stays the primary when it is one of them.
pub(super) fn select_all(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    let nodes = snapshot(world, &state.hierarchy_query);
    let searching = !state.hierarchy_query.trim().is_empty();
    let rows = flatten(ctx, &nodes, searching, state.closed_rows());
    let primary = state.selection.primary();
    let mut listed: Vec<Entity> = rows
        .iter()
        .map(|row| nodes[row.node].entity)
        .filter(|&entity| Some(entity) != primary)
        .collect();
    if primary.is_some_and(|primary| rows.iter().any(|row| nodes[row.node].entity == primary)) {
        listed.extend(primary);
    }
    state.selection.clear();
    for entity in listed {
        state.selection.toggle(entity);
    }
}

fn add_menu(ui: &mut egui::Ui, state: &mut EditorState) {
    // The entries are the Entity menu's own.
    widgets::squircle_command_menu(ui, icons::plus(), "Add entity", |ui| {
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
            id: world.get::<EntityId>(entity).map(|id| *id),
            label: world
                .get::<Name>(entity)
                .map(|name| name.0.clone())
                .unwrap_or_else(|| format!("Entity {}", entity.index())),
            named: world.has::<Name>(entity),
            kind: Kind::of(world, entity),
            children,
            // Roots come first in the order, by construction.
            is_root: index < root_count,
            visible: true,
            prefab: Role::None,
            lock: if world.has::<crate::scene::Locked>(entity) {
                Mark::Own
            } else {
                Mark::Off
            },
            hidden: hidden_mark(world, entity),
            folder: world.get::<crate::scene::Folder>(entity).map(|f| *f),
        })
        .collect();
    // Parents come before children in this order, so one pass carries a lock
    // down every subtree.
    for index in 0..nodes.len() {
        if nodes[index].lock == Mark::Off {
            continue;
        }
        for child in nodes[index].children.clone() {
            if nodes[child].lock == Mark::Off {
                nodes[child].lock = Mark::Inherited;
            }
        }
    }
    roles(world, &mut nodes);

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

/// Whether the editor hides `entity` itself or through an ancestor. The
/// inherited half is read from the derived table rather than walked here,
/// since the renderer has already paid for it.
fn hidden_mark(world: &World, entity: Entity) -> Mark {
    let Some(hidden) = world.get_resource::<crate::scene::EditorHidden>() else {
        return Mark::Off;
    };
    if world
        .get::<orrin_registry::EntityId>(entity)
        .is_some_and(|id| hidden.hides_id(*id))
    {
        Mark::Own
    } else if hidden.contains(entity) {
        Mark::Inherited
    } else {
        Mark::Off
    }
}

/// Mark each node's place in a prefab instance.
///
/// Top-down over `Hierarchy::order`, which puts parents first, so each node
/// inherits the nearest instance above it and is tested against that one
/// alone — one lookup a row, where asking `prefab::instance_of` per row would
/// walk to the root from every one of them.
fn roles(world: &World, nodes: &mut [Node]) {
    // A scene with no instance has no roles to find, and asking costs a
    // component lookup on every row each frame.
    if world.query::<&PrefabInstance>().find(|_, _| true).is_none() {
        return;
    }
    let Some(prefabs) = world.get_resource::<Prefabs>() else {
        return;
    };
    let mut parent = vec![None; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        for &child in &node.children {
            parent[child] = Some(index);
        }
    }
    let mut instance = vec![None; nodes.len()];
    for index in 0..nodes.len() {
        let entity = nodes[index].entity;
        let inherited = parent[index].and_then(|p| instance[p]);
        let id = world.get::<orrin_registry::EntityId>(entity).map(|id| *id);
        if let Some(root) = world.get::<PrefabInstance>(entity)
            && let Some(prefab) = prefabs.get(root.prefab)
        {
            nodes[index].prefab = Role::Root(prefab.name().to_owned());
            instance[index] = id.map(|id| (id, root.prefab));
            continue;
        }
        instance[index] = inherited;
        if let (Some((root, prefab)), Some(id)) = (inherited, id)
            && let Some(prefab) = prefabs.get(prefab)
            && crate::scene::prefab::is_member(prefab, root, id)
        {
            nodes[index].prefab = Role::Member;
        }
    }
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
fn draw_row(
    ui: &mut egui::Ui,
    nodes: &[Node],
    rows: &[Row],
    row: &Row,
    state: &mut EditorState,
    renaming: &mut Option<Renaming>,
) {
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
        let open = row.open;
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
            match node.id {
                Some(id) => state.set_row_open(id, !open),
                None => ui
                    .ctx()
                    .data_mut(|data| data.insert_temp(open_id(node.entity), !open)),
            }
        }
    }
    x += 16.0;
    let glyph = egui::Rect::from_center_size(
        egui::pos2(x + 7.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    let tint = node
        .folder
        .and_then(|folder| folder.tint)
        .map_or(node.kind.tint(weak), |tint| {
            let [r, g, b] = crate::editor::kind::srgb_bytes(tint);
            egui::Color32::from_rgb(r, g, b)
        });
    let tint = if node.hidden != Mark::Off && !selected {
        tint.gamma_multiply(0.45)
    } else {
        tint
    };
    icons::paint(ui, node.kind.icon(), glyph, tint);
    x = glyph.right() + 6.0;

    // The delete affordance sits clear of the floating scroll bar, which
    // appears exactly when the pointer is over the tree — exactly when this
    // button is showing.
    let cross = egui::Rect::from_center_size(
        egui::pos2(rect.right() - scroll_gutter(ui) - 10.0, rect.center().y),
        egui::Vec2::splat(18.0),
    );
    // Every glyph keeps one column whether or not the row is hovered: the
    // cross's is the rightmost, empty at rest, then the lock's, then the
    // eye's. So what is locked, and what is hidden, each reads down one
    // column, and the pointer's arrival moves nothing out from under it.
    let step = |columns: f32| cross.translate(egui::vec2(-20.0 * columns, 0.0));
    let lock_slot = step(1.0);
    let eye_slot = step(2.0);
    let shows_lock = hovered || node.lock != Mark::Off;
    let shows_eye = hovered || node.hidden != Mark::Off;
    let text_right = if shows_eye {
        eye_slot.left() - 4.0
    } else if shows_lock {
        lock_slot.left() - 4.0
    } else {
        rect.right() - widgets::PAD
    };

    let colour = if selected {
        visuals.strong_text_color()
    } else if node.prefab != Role::None {
        theme::PREFAB
    } else {
        visuals.text_color()
    };
    // Dimmed rather than recoloured, so a hidden prefab row still reads as one.
    let colour = if node.hidden != Mark::Off && !selected {
        colour.gamma_multiply(0.45)
    } else {
        colour
    };
    // Coloured in the layout, not at the paint: a plain `WidgetText` bakes the
    // style's text colour into its galley, and the painter's colour then only
    // reaches parts with none — of which there are none.
    let galley = egui::WidgetText::from(egui::RichText::new(&node.label).color(colour))
        .into_galley(
            ui,
            Some(egui::TextWrapMode::Truncate),
            (text_right - x).max(0.0),
            egui::TextStyle::Body,
        );
    // A row whose name was cut is the only one worth a tooltip: repeating a
    // name already fully on screen would pop a bubble over every row the
    // pointer crosses.
    let elided = galley.elided;
    if renaming.as_ref().is_some_and(|r| r.entity == node.entity) {
        let height = ui.spacing().interact_size.y;
        let field = egui::Rect::from_min_max(
            egui::pos2(x - 4.0, rect.center().y - height / 2.0),
            egui::pos2(text_right, rect.center().y + height / 2.0),
        );
        name_field(ui, renaming, field, state);
    } else {
        ui.painter().galley(
            egui::pos2(x, rect.center().y - galley.size().y / 2.0),
            galley,
            colour,
        );
    }
    if !node.children.is_empty() && !hovered {
        ui.painter().text(
            egui::pos2(text_right, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            node.children.len().to_string(),
            egui::TextStyle::Monospace.resolve(ui.style()),
            weak,
        );
    }

    if shows_lock {
        let toggled = toggle(
            ui,
            lock_slot,
            open_id(node.entity).with("lock"),
            node.lock,
            hovered,
            [icons::lock_open(), icons::lock()],
            [
                "Lock: keep it from being picked in the scene",
                "Unlock: let it be picked in the scene again",
            ],
        );
        if toggled {
            if node.lock == Mark::Own {
                state.request_remove_component(node.entity, crate::scene::registry::LOCKED);
            } else {
                state.request_add_component(node.entity, crate::scene::registry::LOCKED);
            }
        }
    }
    if shows_eye {
        let toggled = toggle(
            ui,
            eye_slot,
            open_id(node.entity).with("eye"),
            node.hidden,
            hovered,
            [icons::eye(), icons::eye_slash()],
            [
                "Hide from the editor's view; the game still draws it",
                "Show in the editor's view again",
            ],
        );
        if toggled {
            state.request_toggle_hidden(node.entity);
        }
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

    match (&node.prefab, elided) {
        (Role::Root(prefab), true) => {
            response = response.on_hover_text(format!("{}\nInstance of {prefab}", node.label));
        }
        (Role::Root(prefab), false) => {
            response = response.on_hover_text(format!("Instance of {prefab}"));
        }
        (_, true) => response = response.on_hover_text(&node.label),
        (_, false) => {}
    }
    // The menu acts on the selection, so the row it opened on joins it.
    if response.secondary_clicked() && !selected {
        state.selection.set(node.entity);
    }
    response.context_menu(|ui| row_menu(ui, node, state));
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

/// The name being edited, laid over the row's own: the asset browser's
/// in-place rename (`assets::Cells::name_field`), posted as a name edit.
fn name_field(
    ui: &mut egui::Ui,
    renaming: &mut Option<Renaming>,
    field: egui::Rect,
    state: &mut EditorState,
) {
    let Some(editing) = renaming.as_mut() else {
        return;
    };
    editing.drawn = true;
    // Its own id rather than the row's next: rows come and go with the
    // scroll, and an id that moved would drop the focus.
    let id = egui::Id::new(("hierarchy_rename", editing.entity));
    let output = ui
        .scope_builder(egui::UiBuilder::new().max_rect(field), |ui| {
            egui::TextEdit::singleline(&mut editing.text)
                .id(id)
                .desired_width(field.width())
                .show(ui)
        })
        .inner;
    let response = output.response;
    if !editing.started {
        editing.started = true;
        response.request_focus();
        let mut text = output.state;
        let all = egui::text::CCursorRange::two(
            egui::text::CCursor::new(0),
            egui::text::CCursor::new(editing.text.chars().count()),
        );
        text.cursor.set_char_range(Some(all));
        text.store(ui.ctx(), response.id);
    } else if response.lost_focus() {
        if ui.input(|input| input.key_pressed(egui::Key::Enter)) {
            state.edit(
                editing.entity,
                crate::scene::registry::NAME,
                orrin_registry::Reflect::to_value(&Name::new(editing.text.clone())),
            );
        }
        *renaming = None;
    }
}

/// One of a row's inherited toggles, the lock or the eye: `glyphs` and
/// `hints` are its off and on states. Clickable only on a hovered row and only
/// where the row holds the mark itself, since an inherited one is undone on
/// the row it came from. Reports whether it was clicked.
fn toggle(
    ui: &egui::Ui,
    slot: egui::Rect,
    id: egui::Id,
    mark: Mark,
    hovered: bool,
    glyphs: [icons::Icon; 2],
    hints: [&str; 2],
) -> bool {
    let visuals = ui.visuals();
    let on = mark != Mark::Off;
    let glyph = glyphs[on as usize];
    let resting = match mark {
        Mark::Own => visuals.text_color(),
        Mark::Off | Mark::Inherited => visuals.weak_text_color(),
    };
    if !hovered || mark == Mark::Inherited {
        icons::paint(ui, glyph, slot.shrink(3.0), resting);
        return false;
    }
    let response = ui.interact(slot, id, egui::Sense::click());
    let colour = if response.hovered() {
        visuals.strong_text_color()
    } else {
        resting
    };
    icons::paint(ui, glyph, slot.shrink(3.0), colour);
    response.on_hover_text(hints[on as usize]).clicked()
}

/// What a right-click on a row offers: the Edit menu's commands for the
/// selection, the selection as a new prefab, the whole-instance actions when
/// the row is part of one, and Delete.
fn row_menu(ui: &mut egui::Ui, node: &Node, state: &mut EditorState) {
    selection_commands(ui, state, node.named);
    if let Some(folder) = node.folder {
        tint_menu(ui, node.entity, folder, state);
    }
    if node.prefab != Role::None {
        ui.separator();
        for (label, action) in [
            ("Apply to prefab", PrefabAction::Apply),
            ("Revert to prefab", PrefabAction::Revert),
            ("Unpack", PrefabAction::Unpack),
            ("Show in Assets", PrefabAction::Reveal),
        ] {
            if ui.button(label).clicked() {
                state.request_prefab_action(action, node.entity);
                ui.close_menu();
            }
        }
    }
    ui.separator();
    if ui.button("Delete").clicked() {
        state.request_despawn_selection();
        ui.close_menu();
    }
}

/// The Edit menu's everyday commands for the selection, then the selection
/// as a new prefab: what a row's menu and the inspector's `⋯` both lead
/// with. Rename only for an entity with a name to edit.
pub(super) fn selection_commands(ui: &mut egui::Ui, state: &mut EditorState, named: bool) {
    let item = |ui: &mut egui::Ui, label: &str, key: &str, enabled: bool| {
        let clicked = ui
            .add_enabled(enabled, egui::Button::new(label).shortcut_text(key))
            .clicked();
        if clicked {
            ui.close_menu();
        }
        clicked
    };
    let key = |command| commands::text(ui.ctx(), &state.settings.keys, command);
    let [duplicate, copy, paste, rename] = [
        Command::Duplicate,
        Command::Copy,
        Command::Paste,
        Command::Rename,
    ]
    .map(key);
    if item(ui, "Duplicate", &duplicate, true) {
        state.request_duplicate();
    }
    if item(ui, "Copy", &copy, true) {
        state.request_copy();
    }
    if item(ui, "Paste", &paste, state.can_paste()) {
        state.request_paste();
    }
    if item(ui, "Rename", &rename, named) {
        state.request_rename();
    }
    ui.separator();
    if ui.button("Save as prefab").clicked() {
        state.request_save_prefab();
        ui.close_menu();
    }
}

/// A folder's tint, picked freely, or taken off.
fn tint_menu(
    ui: &mut egui::Ui,
    entity: Entity,
    folder: crate::scene::Folder,
    state: &mut EditorState,
) {
    use crate::editor::kind::{from_srgb_bytes, srgb_bytes};
    use orrin_registry::Reflect;

    ui.separator();
    let mut tinted = folder;
    ui.menu_button("Tint", |ui| {
        let [r, g, b] = folder.tint.map_or([0x4f, 0x8e, 0xe8], srgb_bytes);
        let mut colour = egui::Color32::from_rgb(r, g, b);
        if egui::color_picker::color_picker_color32(
            ui,
            &mut colour,
            egui::color_picker::Alpha::Opaque,
        ) {
            tinted.tint = Some(from_srgb_bytes([colour.r(), colour.g(), colour.b()]));
        }
        if ui
            .add_enabled(folder.tint.is_some(), egui::Button::new("No tint"))
            .clicked()
        {
            tinted.tint = None;
            ui.close_menu();
        }
    });
    if tinted != folder {
        state.edit(entity, crate::scene::registry::FOLDER, tinted.to_value());
    }
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
