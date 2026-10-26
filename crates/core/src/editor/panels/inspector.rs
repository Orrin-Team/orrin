//! Right panel: edit the selected entity. Each section takes a fresh,
//! short-lived borrow of the world so the `RefCell` component storage never
//! double-borrows.
//!
//! Laid out as a property grid: a header naming the entity, a filter, then one
//! collapsible section per component, every row a label in a fixed column and
//! a value filling the rest. The column is what makes a long component list
//! scannable — the eye runs down one edge instead of re-finding each value.

use glam::{EulerRot, Quat, Vec3};

use orrin_ecs::{Entity, World};
use orrin_registry::{Reflect, Registry};

use super::{color_row, figures};
use crate::editor::icons;
use crate::editor::kind::Kind;
use crate::editor::previews::Previews;
use crate::editor::state::EditorState;
use crate::editor::widgets;
#[cfg(feature = "scripting")]
use crate::scene::ScriptComponent;
use crate::scene::prefab;
use crate::scene::{
    AssetPreviews, Assets, Light, LocalTransform, LogBuffer, LogLevel, MaterialHandle, MeshHandle,
    Name, Prefabs, Time, WorldTransform,
};

pub fn body(
    ui: &mut egui::Ui,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
    previews: &mut Previews,
) {
    let Some(entity) = state.selection.primary() else {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(icons::text(
                icons::pointer(),
                20.0,
                ui.visuals().weak_text_color(),
            ));
            ui.weak("Select an entity to inspect it.");
        });
        return;
    };
    if !world.is_alive(entity) {
        state.selection.retain_alive(world);
        return;
    }

    widgets::padded(ui, |ui| {
        header(ui, world, registry, state, entity);
        ui.add_space(4.0);
        widgets::search(ui, &mut state.inspector_query, "Filter properties");
    });
    ui.add_space(6.0);

    let filter = Filter::new(&state.inspector_query);
    // What the selected entity holds differently from its prefab, by
    // component, so each section can say so. Where an instance root stands is
    // its own, not an override worth marking.
    let overridden: Vec<orrin_registry::ComponentId> =
        prefab::member_overrides(world, registry, entity)
            .map(|overrides| {
                overrides
                    .fields
                    .into_iter()
                    .filter(|o| {
                        !(overrides.root == entity
                            && o.component == crate::scene::registry::TRANSFORM)
                    })
                    .map(|o| o.component)
                    .collect()
            })
            .unwrap_or_default();
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            prefab_section(ui, world, registry, state, entity, &filter);
            transform_section(ui, world, state, entity, &filter);
            mesh_section(ui, world, state, entity, previews, &filter);
            material_section(ui, world, entity, previews, &filter);
            light_section(ui, world, state, entity, &filter);
            #[cfg(feature = "scripting")]
            script_section(ui, world, entity, &filter);
            registered_sections(ui, world, registry, state, entity, &filter, &overridden);
            if filter.is_empty() {
                ui.add_space(8.0);
                widgets::padded(ui, |ui| add_component(ui, world, registry, state, entity));
            }
        });
}

/// The inspector's property filter. A section whose title matches shows whole;
/// otherwise it shows only the rows whose label matches, and disappears when
/// none do — so typing "rough" leaves one row on screen rather than every
/// section that happens to contain it somewhere.
struct Filter {
    query: String,
}

impl Filter {
    fn new(query: &str) -> Self {
        Self {
            query: query.trim().to_lowercase(),
        }
    }

    fn is_empty(&self) -> bool {
        self.query.is_empty()
    }

    fn hits(&self, text: &str) -> bool {
        text.to_lowercase().contains(&self.query)
    }

    /// `None` when the section should not draw at all.
    fn section(&self, title: &str, labels: &[&str]) -> Option<Rows<'_>> {
        if self.is_empty() || self.hits(title) {
            return Some(Rows(None));
        }
        labels
            .iter()
            .any(|label| self.hits(label))
            .then_some(Rows(Some(self)))
    }
}

/// Which rows of a shown section draw.
#[derive(Clone, Copy)]
struct Rows<'a>(Option<&'a Filter>);

impl Rows<'_> {
    fn shows(self, label: &str) -> bool {
        self.0.is_none_or(|filter| filter.hits(label))
    }
}

/// The entity's icon, its name as an editable title, what it is and its id,
/// and a menu for what can be done to it as a whole.
fn header(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    let kind = Kind::of(world, entity);
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        let tile = egui::Vec2::splat(32.0);
        let (rect, _) = ui.allocate_exact_size(tile, egui::Sense::hover());
        let tint = kind.tint(weak);
        ui.painter().rect_filled(rect, 5, tint.gamma_multiply(0.16));
        let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(icons::ROW));
        icons::paint(ui, kind.icon(), glyph, tint);

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            widgets::squircle_menu(ui, icons::ellipsis(), "Entity actions", |ui| {
                entity_menu(ui, world, registry, state, entity);
            });

            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                if let Some(mut name) = world.get::<Name>(entity).map(|name| name.clone()) {
                    let renamed = ui
                        .add(
                            egui::TextEdit::singleline(&mut name.0)
                                .font(egui::TextStyle::Heading)
                                .desired_width(f32::INFINITY),
                        )
                        .changed();
                    if renamed {
                        state.edit(entity, crate::scene::registry::NAME, name.to_value());
                    }
                } else {
                    ui.heading(format!("Entity {}", entity.index()));
                }
                let others = match state.selection.len() {
                    0 | 1 => String::new(),
                    n => format!(" · {n} selected"),
                };
                ui.label(
                    figures(format!("{} · id {}{others}", kind.label(), entity.index())).weak(),
                );
            });
        });
    });
}

/// What can be done to an entity as a whole, rather than to one of its
/// components. The header's `⋯` opens it; the panel tests draw it directly.
pub fn entity_menu(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    if ui
        .add(egui::Button::new(icons::with_label(
            icons::terminal(),
            None,
            "Dump to console",
        )))
        .clicked()
    {
        dump(world, registry, entity);
        ui.close_menu();
    }
    ui.separator();
    // The hierarchy deletes the row under the pointer; this deletes the thing
    // being looked at. Both go through the same deferred request — despawning
    // while a panel is iterating the tree is what that defers.
    let name = world.get::<Name>(entity).map_or_else(
        || format!("entity {}", entity.index()),
        |name| name.0.clone(),
    );
    let delete = icons::with_label(icons::trash(), Some(crate::editor::theme::ERROR), "Delete");
    if ui
        .add(egui::Button::new(delete).shortcut_text("Del"))
        .on_hover_text(format!("Despawn {name} and everything under it"))
        .clicked()
    {
        state.request_despawn(entity);
        ui.close_menu();
    }
}

/// Which prefab the entity came from and what it holds differently, each with
/// a way back. Shown on every member of an instance, not only its root: an
/// override belongs to the entity it is on.
fn prefab_section(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(overrides) = prefab::member_overrides(world, registry, entity) else {
        return;
    };
    let (name, path) = world
        .get_resource::<Prefabs>()
        .and_then(|prefabs| {
            let prefab = prefabs.get(overrides.prefab)?;
            Some((prefab.name().to_owned(), prefab.path.clone()))
        })
        .unwrap_or_else(|| (overrides.prefab.to_string(), String::new()));

    // The reserved ids have no vtable to name them.
    let named = |id: &orrin_registry::ComponentId| match id {
        id if *id == crate::scene::registry::PARENT => "Parent".to_owned(),
        id if *id == crate::scene::registry::MESH => "Mesh".to_owned(),
        id if *id == crate::scene::registry::MATERIAL => "Material".to_owned(),
        id => registry
            .get(id)
            .map_or_else(|| id.to_string(), |vtable| vtable.name.to_string()),
    };
    let mut rows: Vec<(
        String,
        String,
        orrin_registry::ComponentId,
        orrin_registry::FieldPath,
    )> = overrides
        .fields
        .iter()
        .map(|o| {
            let what = if o.path.is_empty() {
                "added".to_owned()
            } else {
                o.path.to_string()
            };
            (
                named(&o.component),
                what,
                o.component.clone(),
                o.path.clone(),
            )
        })
        .collect();
    rows.extend(overrides.removed.iter().map(|component| {
        let whole = orrin_registry::FieldPath::empty();
        (
            named(component),
            "removed".to_owned(),
            component.clone(),
            whole,
        )
    }));

    let mut labels: Vec<&str> = rows.iter().map(|(label, ..)| label.as_str()).collect();
    labels.push("Source");
    let Some(shown) = filter.section("Prefab", &labels) else {
        return;
    };

    let mut revert = Vec::new();
    let mut action = None;
    let weak = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with("prefab"),
        icons::box_(),
        weak,
        "Prefab",
        None,
        |ui| {
            if shown.shows("Source") {
                widgets::property(ui, "Source", |ui| ui.label(&name).on_hover_text(&path));
            }
            widgets::padded(ui, |ui| {
                ui.horizontal(|ui| {
                    use crate::editor::state::PrefabAction;
                    for (label, hint, chosen) in [
                        (
                            "Apply",
                            "Write this instance into the prefab, for every instance",
                            PrefabAction::Apply,
                        ),
                        (
                            "Revert all",
                            "Give back everything this instance overrides",
                            PrefabAction::Revert,
                        ),
                        (
                            "Unpack",
                            "Make this instance plain entities",
                            PrefabAction::Unpack,
                        ),
                        ("Show", "Show the prefab in Assets", PrefabAction::Reveal),
                    ] {
                        if ui.button(label).on_hover_text(hint).clicked() {
                            action = Some(chosen);
                        }
                    }
                });
            });
            if rows.is_empty() {
                widgets::padded(ui, |ui| ui.weak("Matches the prefab."));
            }
            for (label, what, component, path) in &rows {
                if !shown.shows(label) {
                    continue;
                }
                widgets::property_with(
                    ui,
                    label,
                    |ui| ui.label(what),
                    |ui| {
                        if icons::button(ui, icons::rotate_ccw(), 12.0, weak, false)
                            .on_hover_text("Revert to the prefab's value")
                            .clicked()
                        {
                            revert.push((component.clone(), path.clone()));
                        }
                    },
                );
            }
        },
    );

    if let Some(action) = action {
        state.request_prefab_action(action, entity);
    }
    for (component, path) in revert {
        state.request_revert(entity, component, path);
    }
}

/// The components above name a Rust type each and draw widgets tuned to it.
/// Anything else the registry knows about is drawn from its vtable instead —
/// which is what makes a component the engine has never heard of, registered by
/// a game assembly after a hot reload, inspectable without a panel being written
/// for it.
const HAND_DRAWN: [orrin_registry::ComponentId; 3] = [
    crate::scene::registry::TRANSFORM,
    crate::scene::registry::NAME,
    crate::scene::registry::LIGHT,
];

/// Draw every registered component that has no section of its own.
///
/// The edit is made on a detached copy and handed to the edit pipeline, which
/// applies the [`diff`] between the two, rather than written straight into
/// storage. That is the discipline architecture §4.4 asks for: one mutation
/// stream, so undo, prefab overrides and eventually sync all see the same
/// changes an inspector drag produced, rather than each having to observe the
/// world for them. The hand-drawn sections above follow it too.
///
/// [`diff`]: orrin_registry::diff
fn registered_sections(
    ui: &mut egui::Ui,
    world: &World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
    overridden: &[orrin_registry::ComponentId],
) {
    for vtable in registry.components() {
        if HAND_DRAWN.contains(&vtable.id) {
            continue;
        }
        let Some(before) = vtable.read(world, entity) else {
            continue;
        };
        let fields: Vec<&str> = match &before {
            orrin_registry::Value::Struct(fields) | orrin_registry::Value::Enum { fields, .. } => {
                fields.iter().map(|(name, _)| name.as_str()).collect()
            }
            _ => Vec::new(),
        };
        if filter.section(&vtable.name, &fields).is_none() {
            continue;
        }

        let mut after = before.clone();
        let tint = if vtable.script.is_some() {
            crate::editor::theme::KIND_SCRIPT
        } else {
            ui.visuals().weak_text_color()
        };
        let icon = if vtable.script.is_some() {
            icons::file_code()
        } else {
            icons::layers()
        };
        let id = ui.id().with(("component", vtable.id.as_str()));
        let mut remove = false;
        let mut menu = |ui: &mut egui::Ui| {
            if ui.button("Remove component").clicked() {
                remove = true;
                ui.close_menu();
            }
        };
        let title = overridden
            .contains(&vtable.id)
            .then_some(crate::editor::theme::PREFAB);
        let touched = widgets::section_titled(
            ui,
            id,
            icon,
            tint,
            &vtable.name,
            title,
            Some(&mut menu),
            |ui| (vtable.inspect)(ui, &mut after),
        )
        .unwrap_or(false);
        if remove {
            state.request_remove_component(entity, vtable.id.clone());
        } else if touched {
            state.edit(entity, vtable.id.clone(), after);
        }
    }
}

/// Every registered component the entity does not have yet, at its default.
/// Hand-drawn ones are offered too: a bare entity gaining a transform is the
/// commonest addition of all.
fn add_component(
    ui: &mut egui::Ui,
    world: &World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    let missing: Vec<(orrin_registry::ComponentId, String)> = registry
        .components()
        .filter(|vtable| !vtable.has(world, entity))
        .map(|vtable| (vtable.id.clone(), vtable.name.to_string()))
        .collect();

    let button = egui::Button::new(icons::with_label(icons::plus(), None, "Add component"))
        .min_size(egui::vec2(ui.available_width(), 28.0));
    let response = ui.add_enabled(!missing.is_empty(), button);
    let popup = ui.make_persistent_id("add_component");
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }
    egui::popup_below_widget(
        ui,
        popup,
        &response,
        egui::PopupCloseBehavior::CloseOnClick,
        |ui| {
            ui.set_min_width(response.rect.width() - 12.0);
            add_component_entries(ui, state, entity, &missing);
        },
    );
}

fn add_component_entries(
    ui: &mut egui::Ui,
    state: &mut EditorState,
    entity: Entity,
    missing: &[(orrin_registry::ComponentId, String)],
) {
    for (id, name) in missing {
        if ui
            .add(egui::Button::new(name.as_str()).shortcut_text(id.as_str()))
            .clicked()
        {
            state.request_add_component(entity, id.clone());
        }
    }
}

/// Print the entity exactly as the registry sees it.
///
/// Every section above names a concrete component type; this one names none —
/// it asks each registered type whether it's present and prints whatever comes
/// back. That difference is the point of the registry, and this is how a
/// component's registration gets eyeballed.
pub(super) fn dump(world: &mut World, registry: &Registry, entity: Entity) {
    let mut text = String::new();
    orrin_registry::write_entity(&mut text, registry, world, entity);

    // Both resource borrows are taken after the dump is built, so no component
    // borrow is still open when the log is written.
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.push(LogLevel::Info, text, frame);
    }
}

/// A transform row: three axis fields, then a reset that only appears once
/// the value has left its default — so a glance down the column shows which
/// of the three have been touched.
fn transform_row(
    ui: &mut egui::Ui,
    label: &str,
    values: &mut [f32; 3],
    default: [f32; 3],
    speed: f64,
    suffix: &str,
) -> bool {
    let moved = values
        .iter()
        .zip(default)
        .any(|(value, default)| (value - default).abs() > 1e-4);
    let mut reset = false;
    let [x, y, z] = values;
    let mut changed = widgets::property_with(
        ui,
        label,
        |ui| widgets::axes(ui, [x, y, z], speed, suffix),
        |ui| {
            if moved {
                let weak = ui.visuals().weak_text_color();
                reset = icons::button(ui, icons::rotate_ccw(), 12.0, weak, false)
                    .on_hover_text(format!("Reset {}", label.to_lowercase()))
                    .clicked();
            }
        },
    );
    if reset {
        *values = default;
        changed = true;
    }
    changed
}

fn transform_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(rows) = filter.section("Transform", &["Position", "Rotation", "Scale", "World"])
    else {
        return;
    };
    let Some(mut t) = world.get::<LocalTransform>(entity).map(|t| *t) else {
        return;
    };
    let mut edited = false;
    // Read once for the rows; the menu below is what may change it.
    let uniform_scale = state.scale_linked;
    let linked = &mut state.scale_linked;
    let mut reset = false;
    let mut menu = |ui: &mut egui::Ui| {
        if ui.button("Reset transform").clicked() {
            reset = true;
            ui.close_menu();
        }
        ui.checkbox(linked, "Uniform scale");
    };
    let id = ui.id().with("transform");
    widgets::section(
        ui,
        id,
        icons::move_(),
        ui.visuals().weak_text_color(),
        "Transform",
        Some(&mut menu),
        |ui| {
            if rows.shows("Position") {
                let mut position = t.translation.to_array();
                if transform_row(ui, "Position", &mut position, [0.0; 3], 0.05, "") {
                    t.translation = Vec3::from(position);
                    edited = true;
                }
            }

            // Edited as Euler degrees in the same convention as the C# scripting API
            // (`Quaternion.Euler` / `eulerAngles`): Z-X-Y application, i.e. glam's
            // intrinsic `EulerRot::YXZ`. Matching it means the numbers shown here equal
            // a script's `eulerAngles` for the same rotation. Mind the reordering: YXZ
            // takes/yields angles as (yaw=Y, pitch=X, roll=Z), but the three fields are
            // laid out X, Y, Z (pitch, yaw, roll).
            if rows.shows("Rotation") {
                let (yaw, pitch, roll) = t.rotation.to_euler(EulerRot::YXZ);
                // `to_euler` hands back -0.0, or a crumb below it, for an
                // unrotated axis, which reads as "-0.00°" on every fresh entity.
                let mut euler = [pitch, yaw, roll].map(|angle| {
                    let degrees = angle.to_degrees();
                    if degrees.abs() < 1e-4 { 0.0 } else { degrees }
                });
                if transform_row(ui, "Rotation", &mut euler, [0.0; 3], 0.5, "°") {
                    t.rotation = Quat::from_euler(
                        EulerRot::YXZ,
                        euler[1].to_radians(), // yaw   (Y)
                        euler[0].to_radians(), // pitch (X)
                        euler[2].to_radians(), // roll  (Z)
                    );
                    edited = true;
                }
            }

            if rows.shows("Scale") {
                let before = t.scale.to_array();
                let mut scale = before;
                let label = if uniform_scale {
                    "Scale (uniform)"
                } else {
                    "Scale"
                };
                if transform_row(ui, label, &mut scale, [1.0; 3], 0.05, "") {
                    t.scale = Vec3::from(if uniform_scale {
                        uniform(before, scale)
                    } else {
                        scale
                    });
                    edited = true;
                }
            }

            // The fields above are parent-relative. For anything with a parent
            // that makes them baffling on their own — a cube sitting visibly at
            // x=12 reads "Position 0". The world position is the reconciling
            // number, and it is read-only because it is derived.
            if rows.shows("World")
                && !crate::scene::is_transform_root(world, entity)
                && let Some(world_transform) = world.get::<WorldTransform>(entity)
            {
                let position = world_transform.translation();
                widgets::property(ui, "World", |ui| {
                    ui.label(
                        figures(format!(
                            "{:.2}, {:.2}, {:.2}",
                            position.x, position.y, position.z
                        ))
                        .weak(),
                    );
                });
            }
        },
    );
    if reset {
        t = LocalTransform::default();
        edited = true;
    }
    if edited {
        state.edit(entity, crate::scene::registry::TRANSFORM, t.to_value());
    }
}

/// A uniform-scale edit: whichever axis moved sets the ratio for all three.
/// An axis at zero has no ratio to carry, so the new value is copied instead.
fn uniform(before: [f32; 3], after: [f32; 3]) -> [f32; 3] {
    let Some(axis) = (0..3).find(|&i| before[i] != after[i]) else {
        return after;
    };
    if before[axis] == 0.0 {
        return [after[axis]; 3];
    }
    let ratio = after[axis] / before[axis];
    before.map(|value| value * ratio)
}

fn mesh_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    previews: &mut Previews,
    filter: &Filter,
) {
    if !world.has::<MeshHandle>(entity) {
        return;
    }
    let Some(rows) = filter.section("Mesh renderer", &["Mesh", "Material"]) else {
        return;
    };
    let id = ui.id().with("mesh_renderer");
    let mut picked_mesh = None;
    let mut picked_material = None;
    widgets::section(
        ui,
        id,
        icons::box_(),
        crate::editor::theme::KIND_MESH,
        "Mesh renderer",
        None,
        |ui| {
            if rows.shows("Mesh") {
                widgets::property(ui, "Mesh", |ui| {
                    picked_mesh = mesh_picker(ui, world, entity, previews);
                });
            }
            if rows.shows("Material") && world.has::<MaterialHandle>(entity) {
                widgets::property(ui, "Material", |ui| {
                    picked_material = material_picker(ui, world, entity, previews);
                });
            }
        },
    );
    // Posted by name, the form a scene file holds them in, for the reason
    // `persist::asset_refs` gives.
    let assets = world.resource::<Assets>();
    if let Some(name) = picked_mesh.and_then(|handle| assets.mesh_name(handle)) {
        let value = orrin_registry::Value::String(name.to_owned());
        state.edit(entity, crate::scene::registry::MESH, value);
    }
    if let Some(name) = picked_material.and_then(|handle| assets.material_name(handle)) {
        let value = orrin_registry::Value::String(name.to_owned());
        state.edit(entity, crate::scene::registry::MATERIAL, value);
    }
}

/// One choice in a picker: the asset's name, its handle, and the text the
/// filter matches against.
struct Choice<H> {
    name: String,
    handle: H,
}

/// The slot, and below it a filterable list of every asset of its type.
///
/// The list is drawn with `show_rows`, so a model that brings a few thousand
/// meshes costs the rows on screen and not the rest. Returns the chosen
/// handle if it differs from `current`; all world borrows are dropped by then,
/// so the caller can safely `insert`.
fn picker<H: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    choices: Vec<Choice<H>>,
    current: Option<H>,
    slot: impl FnOnce(&mut egui::Ui, &str) -> egui::Response,
    thumbnail: &mut dyn FnMut(&egui::Context, H) -> Option<egui::TextureHandle>,
) -> Option<H> {
    let name = current
        .and_then(|h| choices.iter().find(|choice| choice.handle == h))
        .map_or("—", |choice| choice.name.as_str())
        .to_owned();
    let response = slot(ui, &name);
    let popup = ui.make_persistent_id(("picker", id));
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }

    let mut chosen = None;
    egui::popup_below_widget(
        ui,
        popup,
        &response,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_width(response.rect.width().max(240.0));
            let query_id = popup.with("query");
            let mut query: String = ui.data_mut(|data| data.get_temp(query_id).unwrap_or_default());
            widgets::search(ui, &mut query, "Filter");
            let needle = query.trim().to_lowercase();
            let shown: Vec<&Choice<H>> = choices
                .iter()
                .filter(|choice| needle.is_empty() || choice.name.to_lowercase().contains(&needle))
                .collect();
            ui.data_mut(|data| data.insert_temp(query_id, query));

            const ROW: f32 = 28.0;
            egui::ScrollArea::vertical().max_height(320.0).show_rows(
                ui,
                ROW,
                shown.len(),
                |ui, range| {
                    for choice in &shown[range] {
                        let selected = Some(choice.handle) == current;
                        let response = ui
                            .horizontal(|ui| {
                                ui.set_min_height(ROW - ui.spacing().item_spacing.y);
                                let size = egui::Vec2::splat(ROW - 6.0);
                                match thumbnail(ui.ctx(), choice.handle) {
                                    Some(texture) => {
                                        ui.add(egui::Image::new((texture.id(), size)));
                                    }
                                    None => {
                                        ui.add_space(size.x + ui.spacing().item_spacing.x);
                                    }
                                }
                                ui.add(
                                    egui::Button::new(choice.name.as_str())
                                        .selected(selected)
                                        .frame(selected)
                                        .min_size(egui::vec2(ui.available_width(), 0.0)),
                                )
                            })
                            .inner;
                        if response.clicked() {
                            chosen = Some(choice.handle);
                            ui.memory_mut(|memory| memory.close_popup());
                        }
                    }
                },
            );
        },
    );

    chosen.filter(|&handle| Some(handle) != current)
}

fn mesh_picker(
    ui: &mut egui::Ui,
    world: &World,
    entity: Entity,
    previews: &mut Previews,
) -> Option<MeshHandle> {
    let mut choices: Vec<Choice<MeshHandle>> = world
        .resource::<Assets>()
        .meshes()
        .map(|(name, handle)| Choice {
            name: name.to_owned(),
            handle,
        })
        .collect();
    choices.sort_by(|a, b| a.name.cmp(&b.name));
    let current = world.get::<MeshHandle>(entity).map(|h| *h);
    let texture = current.and_then(|handle| previews.mesh(ui.ctx(), world, handle));
    let subtitle = current
        .and_then(|handle| previews.mesh_summary(world, handle))
        .unwrap_or_default();

    picker(
        ui,
        "mesh",
        choices,
        current,
        |ui, name| widgets::asset_slot(ui, texture.as_ref(), icons::box_(), name, &subtitle),
        &mut |ctx, handle| previews.mesh(ctx, world, handle),
    )
}

fn material_picker(
    ui: &mut egui::Ui,
    world: &World,
    entity: Entity,
    previews: &mut Previews,
) -> Option<MaterialHandle> {
    let mut choices: Vec<Choice<MaterialHandle>> = world
        .resource::<Assets>()
        .materials()
        .map(|(name, handle)| Choice {
            name: name.to_owned(),
            handle,
        })
        .collect();
    choices.sort_by(|a, b| a.name.cmp(&b.name));
    let current = world.get::<MaterialHandle>(entity).map(|h| *h);
    let texture = current.and_then(|handle| previews.material(ui.ctx(), world, handle));
    let subtitle = current
        .and_then(|handle| previews.material_summary(world, handle))
        .unwrap_or_default();

    picker(
        ui,
        "material",
        choices,
        current,
        |ui, name| widgets::asset_slot(ui, texture.as_ref(), icons::palette(), name, &subtitle),
        &mut |ctx, handle| previews.material(ctx, world, handle),
    )
}

/// The selected entity's material, edited in place.
///
/// A material is shared by every entity that draws with it, so an edit here
/// changes all of them — which is what editing a material means. The edit goes
/// to the world-side copy and is queued for the renderer (see
/// [`AssetPreviews`]), never written into a component.
fn material_section(
    ui: &mut egui::Ui,
    world: &World,
    entity: Entity,
    previews: &mut Previews,
    filter: &Filter,
) {
    let Some(handle) = world.get::<MaterialHandle>(entity).map(|h| *h) else {
        return;
    };
    let Some((before, textured)) = world
        .get_resource::<AssetPreviews>()
        .and_then(|kept| kept.material(handle).map(|p| (p.material, p.textured)))
    else {
        return;
    };
    let labels = ["Base colour", "Metallic", "Roughness", "Emission", "Blend"];
    let Some(rows) = filter.section("Material", &labels) else {
        return;
    };

    let mut after = before;
    let id = ui.id().with("material");
    let weak = ui.visuals().weak_text_color();
    widgets::section(ui, id, icons::palette(), weak, "Material", None, |ui| {
        if rows.shows("Base colour") {
            widgets::property(ui, "Base colour", |ui| {
                let mut rgb = after.base_color.to_array();
                ui.color_edit_button_rgb(&mut rgb);
                after.base_color = Vec3::from(rgb);
                if textured {
                    ui.label(figures("× texture").weak());
                }
            });
        }
        if rows.shows("Metallic") {
            widgets::property(ui, "Metallic", |ui| {
                ui.add(egui::Slider::new(&mut after.metallic, 0.0..=1.0).fixed_decimals(2));
            });
        }
        if rows.shows("Roughness") {
            widgets::property(ui, "Roughness", |ui| {
                ui.add(egui::Slider::new(&mut after.roughness, 0.0..=1.0).fixed_decimals(2));
            });
        }
        if rows.shows("Emission") {
            // Edited as a colour and a luminance rather than three nits: the
            // hue and the brightness are the two things a person adjusts, and
            // three unbounded channels make either one a chore.
            let nits = after.emissive.max_element();
            let mut tint = if nits > 0.0 {
                (after.emissive / nits).to_array()
            } else {
                [1.0; 3]
            };
            let mut luminance = nits;
            widgets::property(ui, "Emission", |ui| {
                ui.color_edit_button_rgb(&mut tint);
                // The swatch takes its share of the row, so the slider gets
                // what is left rather than the whole row's width.
                let swatch = ui.spacing().interact_size.x + ui.spacing().item_spacing.x;
                ui.spacing_mut().slider_width = (ui.spacing().slider_width - swatch).max(24.0);
                ui.add(
                    egui::Slider::new(&mut luminance, 0.0..=50_000.0)
                        .logarithmic(true)
                        .max_decimals(0)
                        .suffix(" nt"),
                )
                .on_hover_text("Luminance in cd/m² (nits)");
            });
            after.emissive = Vec3::from(tint) * luminance;
        }
        if rows.shows("Blend") {
            let blend = previews
                .material_info(world, handle)
                .map_or("—", |info| info.blend);
            widgets::property(ui, "Blend", |ui| ui.label(blend));
        }
    });

    let edited = before.base_color != after.base_color
        || before.metallic != after.metallic
        || before.roughness != after.roughness
        || before.emissive != after.emissive;
    if edited && let Some(mut kept) = world.get_resource_mut::<AssetPreviews>() {
        kept.edit_material(handle, |material| *material = after);
        previews.forget_material(handle);
    }
}

#[cfg(feature = "scripting")]
fn script_section(ui: &mut egui::Ui, world: &World, entity: Entity, filter: &Filter) {
    let Some(mut script) = world.get_mut::<ScriptComponent>(entity) else {
        return;
    };
    let Some(rows) = filter.section("Script", &["Behaviour", "Enabled", "State"]) else {
        return;
    };
    let id = ui.id().with("script");
    widgets::section(
        ui,
        id,
        icons::file_code(),
        crate::editor::theme::KIND_SCRIPT,
        "Script",
        None,
        |ui| {
            if rows.shows("Behaviour") {
                let type_name = script.type_name.clone();
                widgets::property(ui, "Behaviour", |ui| ui.label(figures(type_name)));
            }
            if rows.shows("Enabled") {
                // Only the desired state is written here; the script tick sees
                // the change next frame and dispatches OnEnable/OnDisable itself.
                widgets::property(ui, "Enabled", |ui| ui.checkbox(&mut script.enabled, ""));
            }
            if rows.shows("State") {
                widgets::property(ui, "State", |ui| {
                    if script.faulted {
                        // A hook threw and the engine disabled the script (see
                        // the log for the exception). Clearing re-arms it from
                        // next tick — the manual counterpart to "re-enable on
                        // hot reload".
                        icons::labelled(
                            ui,
                            icons::warning(),
                            crate::editor::theme::ERROR,
                            "faulted",
                        );
                        if ui.small_button("Clear fault").clicked() {
                            script.faulted = false;
                        }
                    } else {
                        ui.weak(match (script.active, script.started) {
                            (true, _) => "active",
                            (false, true) => "inactive",
                            (false, false) => "not started",
                        });
                    }
                });
            }
        },
    );
}

/// Range is where the light is cut off, not how far it carries — inverse-square
/// decides the latter and never reaches zero. Worth spelling out on the tooltip
/// now that brightness is a physical quantity: the instinct with a unitless
/// intensity was to reach for this slider to make a light dimmer, and it is the
/// one control here that cannot do that.
fn range_row(ui: &mut egui::Ui, range: &mut f32) {
    widgets::property(ui, "Range", |ui| {
        ui.add(egui::Slider::new(range, 0.0..=100.0).suffix(" m"))
            .on_hover_text(
                "Where the falloff is clamped to zero, so this light stops costing anything \
                 past it. A budget, not a brightness.",
            );
    });
}

fn power_row(ui: &mut egui::Ui, lumens: &mut f32, hint: &str) {
    widgets::property(ui, "Power", |ui| {
        ui.add(
            egui::Slider::new(lumens, 1.0..=100_000.0)
                .logarithmic(true)
                .suffix(" lm"),
        )
        .on_hover_text(hint);
    });
}

fn shadows_row(ui: &mut egui::Ui, casts_shadows: &mut bool, hint: &str) {
    widgets::property(ui, "Shadows", |ui| {
        ui.checkbox(casts_shadows, "Cast").on_hover_text(hint);
    });
}

fn light_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<Light>(entity).map(|light| *light) else {
        return;
    };
    let labels = [
        "Colour",
        "Illuminance",
        "Power",
        "Range",
        "Reflector",
        "Outer angle",
        "Inner angle",
        "Shadows",
    ];
    if filter.section("Light", &labels).is_none() {
        return;
    }
    let mut light = before;
    let (icon, title) = match light {
        Light::Directional { .. } => (icons::sun(), "Sun"),
        Light::Point { .. } => (icons::lightbulb(), "Point light"),
        Light::Spot { .. } => (icons::lightbulb(), "Spot light"),
    };
    let id = ui.id().with("light");
    widgets::section(
        ui,
        id,
        icon,
        crate::editor::theme::KIND_LIGHT,
        title,
        None,
        |ui| match &mut light {
            Light::Directional { color, lux } => {
                color_row(ui, "Colour", color);
                // Logarithmic, as every one of these is: the useful range spans
                // five orders of magnitude, and on a linear slider everything
                // dimmer than an overcast day is the leftmost pixel.
                widgets::property(ui, "Illuminance", |ui| {
                    ui.add(
                        egui::Slider::new(lux, 1.0..=150_000.0)
                            .logarithmic(true)
                            .suffix(" lx"),
                    )
                    .on_hover_text("Noon sun 100 000, overcast 20 000, sunset under 1 000");
                });
            }
            Light::Point {
                color,
                lumens,
                range,
                casts_shadows,
            } => {
                color_row(ui, "Colour", color);
                power_row(
                    ui,
                    lumens,
                    "Domestic bulb 800, shop light 4 000, flood tens of thousands",
                );
                range_row(ui, range);
                shadows_row(
                    ui,
                    casts_shadows,
                    "Six atlas tiles, and only if the budget reaches this light",
                );
            }
            Light::Spot {
                color,
                lumens,
                range,
                inner_angle,
                outer_angle,
                reflector,
                casts_shadows,
            } => {
                color_row(ui, "Colour", color);
                power_row(
                    ui,
                    lumens,
                    "The bulb's own output, before the cone concentrates it",
                );
                widgets::property(ui, "Reflector", |ui| {
                    ui.checkbox(reflector, "").on_hover_text(
                        "On, the cone concentrates the power and narrowing it brightens the beam. \
                         Off, the cone only masks a bare bulb and the angle is free of brightness.",
                    );
                });
                range_row(ui, range);
                // Dragged together: the inner cone cannot leave the outer one,
                // and the falloff divides by their difference — so clamping
                // here is what keeps the editor from authoring a light the
                // shader has to defend itself against.
                widgets::property(ui, "Outer angle", |ui| {
                    ui.add(egui::Slider::new(outer_angle, 1.0..=89.0).suffix("°"))
                        .on_hover_text("Half angle from the axis, where the cone reaches nothing");
                });
                *inner_angle = inner_angle.min(*outer_angle);
                widgets::property(ui, "Inner angle", |ui| {
                    ui.add(egui::Slider::new(inner_angle, 0.0..=*outer_angle).suffix("°"))
                        .on_hover_text("Half angle out to which it is still at full brightness");
                });
                shadows_row(
                    ui,
                    casts_shadows,
                    "One atlas tile, and only if the budget reaches this light",
                );
            }
        },
    );
    let after = light.to_value();
    if !orrin_registry::diff(&before.to_value(), &after).is_empty() {
        state.edit(entity, crate::scene::registry::LIGHT, after);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Collider, ColliderShape, Spin, Tag, register_components};
    use orrin_registry::Value;

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    }

    /// A hand-written section skips its component here by id. Rename the id and
    /// the skip stops matching, so the component is drawn twice — once tuned and
    /// once generic — which looks like a duplicated panel and reads as a
    /// rendering bug rather than as the rename it is.
    #[test]
    fn every_hand_drawn_component_is_registered_under_the_id_it_is_skipped_by() {
        let registry = registry();
        for id in &HAND_DRAWN {
            assert!(
                registry.get(id).is_some(),
                "`{id}` is skipped by the inspector but not registered"
            );
        }
    }

    /// The other side of the same coin: every component the registry knows and
    /// no section draws must reach the generic inspector. Named here so that
    /// adding a component without a section is visibly a decision.
    #[test]
    fn everything_else_is_drawn_from_its_vtable() {
        let registry = registry();
        let generic: Vec<&str> = registry
            .components()
            .filter(|c| !HAND_DRAWN.contains(&c.id))
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(
            generic,
            ["orrin.tag", "orrin.collider", "orrin.spin"],
            "registration order changed, or a component gained/lost a section"
        );
    }

    /// The path an inspector drag takes: read the component, edit the detached
    /// value, diff, apply. Driven here without egui, because what is worth
    /// asserting is that the change survives the round trip onto a real
    /// component — including one whose shape is an enum.
    #[test]
    fn an_edit_travels_as_a_diff_and_lands_on_the_component() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(
            entity,
            Collider {
                shape: ColliderShape::Sphere { radius: 1.0 },
                is_trigger: false,
            },
        );

        let vtable = registry.get(&crate::scene::registry::COLLIDER).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let mut after = before.clone();
        *after.field_mut("is_trigger").unwrap() = Value::Bool(true);
        *after
            .field_mut("shape")
            .unwrap()
            .field_mut("radius")
            .unwrap() = Value::F32(2.5);

        let changes = orrin_registry::diff(&before, &after);
        assert_eq!(changes.len(), 2);
        vtable.apply(&mut world, entity, &changes).unwrap();

        let collider = world.get::<Collider>(entity).unwrap();
        assert!(collider.is_trigger);
        assert!(matches!(collider.shape, ColliderShape::Sphere { radius } if radius == 2.5));
    }

    /// `Spin::from_value` refuses an axis it cannot normalize. That refusal has
    /// to survive the diff route as well, or an inspector drag becomes the one
    /// way into the world that skips a component's own validation.
    #[test]
    fn a_component_still_refuses_a_value_its_constructor_rejects() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Spin::new(Vec3::Y, 1.0));

        let vtable = registry.get(&crate::scene::registry::SPIN).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let mut after = before.clone();
        *after.field_mut("axis").unwrap() = Value::Vec3(Vec3::ZERO);

        let error = vtable
            .apply(&mut world, entity, &orrin_registry::diff(&before, &after))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "field `axis`: expected a non-zero axis, found Vec3(0.0, 0.0, 0.0)"
        );
        assert_eq!(vtable.read(&world, entity), Some(before));
    }

    /// A hand-written section is the exception, so the generic one has to handle
    /// a component it has never been told about — the case every game assembly
    /// component is in.
    #[test]
    fn a_component_with_no_section_reads_back_what_the_inspector_drew() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Tag::new("player"));

        let vtable = registry.get(&crate::scene::registry::TAG).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let drawn = std::cell::RefCell::new(before.clone());
        egui::__run_test_ui(|ui| {
            assert!(!(vtable.inspect)(ui, &mut drawn.borrow_mut()));
        });
        assert_eq!(drawn.into_inner(), before);
    }

    #[test]
    fn a_filter_keeps_a_whole_section_when_its_title_matches() {
        let filter = Filter::new("trans");
        let rows = filter.section("Transform", &["Position"]).unwrap();
        assert!(rows.shows("Scale"));
    }

    #[test]
    fn a_filter_keeps_only_matching_rows_otherwise() {
        let filter = Filter::new("ROUGH");
        assert!(
            filter
                .section("Transform", &["Position", "Scale"])
                .is_none()
        );
        let rows = filter
            .section("Material", &["Metallic", "Roughness"])
            .unwrap();
        assert!(rows.shows("Roughness"));
        assert!(!rows.shows("Metallic"));
    }

    #[test]
    fn uniform_scale_carries_the_moved_axis_ratio_to_all_three() {
        assert_eq!(uniform([1.0, 2.0, 4.0], [1.0, 3.0, 4.0]), [1.5, 3.0, 6.0]);
        assert_eq!(uniform([0.0, 1.0, 1.0], [2.0, 1.0, 1.0]), [2.0, 2.0, 2.0]);
        assert_eq!(uniform([1.0; 3], [1.0; 3]), [1.0; 3]);
    }

    /// Adding goes through the same deferred request every other structural
    /// edit does, and lands the component's registered default.
    #[test]
    fn adding_a_component_puts_its_default_on_the_entity() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        let mut state = EditorState::default();

        state.request_add_component(entity, crate::scene::registry::TAG);
        state.apply(&mut world, &registry);

        let vtable = registry.get(&crate::scene::registry::TAG).unwrap();
        assert_eq!(vtable.read(&world, entity), Some(vtable.default()));
    }
}
