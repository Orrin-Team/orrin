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
use orrin_registry::{AssetId, Reflect, Registry};

use super::{color_row, figures};
use crate::asset_db::AssetDatabase;
use crate::editor::catalog;
use crate::editor::icons;
use crate::editor::kind::Kind;
use crate::editor::previews::Previews;
use crate::editor::state::EditorState;
use crate::editor::widgets;
use crate::gfx::vulkan::material_preview;
#[cfg(feature = "scripting")]
use crate::scene::Scripts;
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
            material_section(ui, world, state, entity, previews, &filter);
            light_section(ui, world, state, entity, &filter);
            physics_section(ui, world, state, entity, &filter);
            audio_section(ui, world, state, entity, &filter, &overridden);
            super::ui_sections::sections(ui, world, state, entity, &filter);
            #[cfg(feature = "scripting")]
            script_sections(ui, world, registry, state, entity, &filter, &overridden);
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
pub(super) struct Filter {
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
    pub(super) fn section(&self, title: &str, labels: &[&str]) -> Option<Rows<'_>> {
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
pub(super) struct Rows<'a>(Option<&'a Filter>);

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
const HAND_DRAWN: [orrin_registry::ComponentId; 17] = [
    crate::scene::registry::TRANSFORM,
    crate::scene::registry::NAME,
    crate::scene::registry::LIGHT,
    crate::scene::registry::MESH,
    crate::scene::registry::MATERIAL,
    crate::scene::registry::CANVAS,
    crate::scene::registry::RECT_TRANSFORM,
    crate::scene::registry::IMAGE,
    crate::scene::registry::TEXT,
    crate::scene::registry::BUTTON,
    crate::scene::registry::GRADIENT,
    crate::scene::registry::CORNER,
    crate::scene::registry::STROKE,
    crate::scene::registry::SHADOW,
    crate::scene::registry::SCROLL_FRAME,
    crate::scene::registry::PHYSICS_SETTINGS,
    crate::scene::registry::AUDIO_SOURCE,
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
        // A Behaviour's fields are drawn in its own script section.
        if HAND_DRAWN.contains(&vtable.id) || vtable.script.is_some() {
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
        let tint = ui.visuals().weak_text_color();
        let (_, icon) = catalog::describe(vtable);
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

/// Every registered component the entity does not have yet, at its default,
/// filed under its group with its icon. Hand-drawn ones are offered too: a bare
/// entity gaining a transform is the commonest addition of all.
fn add_component(
    ui: &mut egui::Ui,
    world: &World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    let mut missing: Vec<Offer> = registry
        .components()
        .filter(|vtable| !vtable.has(world, entity))
        .map(|vtable| {
            let (group, icon) = catalog::describe(vtable);
            Offer {
                id: vtable.id.clone(),
                name: vtable.name.to_string(),
                group,
                icon,
            }
        })
        .collect();
    // Stable, so a group keeps the registry's order: a canvas before what goes
    // on one.
    missing.sort_by_key(|offer| offer.group);

    let button = egui::Button::new(icons::with_label(icons::plus(), None, "Add component"))
        .min_size(egui::vec2(ui.available_width(), 28.0));
    let response = ui.add_enabled(!missing.is_empty(), button);
    let popup = egui::Id::new("add_component");
    let query_id = popup.with("query");
    let at_id = popup.with("at");
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
        let at = response
            .interact_pointer_pos()
            .unwrap_or(response.rect.left_bottom());
        ui.data_mut(|data| {
            data.remove::<String>(query_id);
            data.insert_temp(at_id, at);
        });
    }
    if !ui.memory(|memory| memory.is_popup_open(popup)) {
        return;
    }
    // An area of its own rather than `popup_below_widget`, which holds a popup
    // to at least the width of the button that opened it — here the panel's —
    // and opens it under the button rather than under the pointer.
    let at: egui::Pos2 = ui
        .data(|data| data.get_temp(at_id))
        .unwrap_or(response.rect.left_bottom());
    let area = egui::Area::new(popup)
        .kind(egui::UiKind::Popup)
        .order(egui::Order::Foreground)
        .fixed_pos(at + egui::vec2(0.0, 4.0))
        .pivot(egui::Align2::LEFT_TOP)
        .constrain(true)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .show(ui, |ui| {
                    ui.set_width(ADD_MENU_WIDTH);
                    let mut query: String = ui
                        .data_mut(|data| data.get_temp(query_id))
                        .unwrap_or_default();
                    widgets::search(ui, &mut query, "Search components").request_focus();
                    let enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
                    ui.data_mut(|data| data.insert_temp(query_id, query.clone()));

                    let needle = query.trim().to_lowercase();
                    let shown: Vec<&Offer> = missing
                        .iter()
                        .filter(|offer| {
                            needle.is_empty() || offer.name.to_lowercase().contains(&needle)
                        })
                        .collect();
                    if enter {
                        return shown.first().map(|offer| offer.id.clone());
                    }
                    let height = ADD_MENU_HEIGHT.min(ui.ctx().screen_rect().height() - 64.0);
                    // Floored as well as capped: an area keeps the size its first
                    // frame measured, and that is egui's 400px default.
                    egui::ScrollArea::vertical()
                        .max_height(height)
                        .min_scrolled_height(height)
                        .show(ui, |ui| add_component_entries(ui, &shown))
                        .inner
                })
                .inner
        });
    let escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
    let outside = response.clicked_elsewhere() && area.response.clicked_elsewhere();
    if let Some(id) = area.inner {
        state.request_add_component(entity, id);
        ui.memory_mut(|memory| memory.close_popup());
    } else if escape || outside {
        ui.memory_mut(|memory| memory.close_popup());
    }
}

const ADD_MENU_WIDTH: f32 = 220.0;
const ADD_MENU_HEIGHT: f32 = 560.0;

struct Offer {
    id: orrin_registry::ComponentId,
    name: String,
    group: catalog::Group,
    icon: icons::Icon,
}

/// The rows under their headings; returns the one clicked.
fn add_component_entries(
    ui: &mut egui::Ui,
    shown: &[&Offer],
) -> Option<orrin_registry::ComponentId> {
    if shown.is_empty() {
        ui.weak("No component matches");
        return None;
    }
    let weak = ui.visuals().weak_text_color();
    ui.spacing_mut().item_spacing.y = 0.0;
    ui.spacing_mut().button_padding.y = 3.0;
    // Bare until hovered, the way egui styles a menu's rows.
    let inactive = &mut ui.visuals_mut().widgets.inactive;
    inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
    inactive.bg_stroke = egui::Stroke::NONE;
    let mut chosen = None;
    let mut group = None;
    for offer in shown {
        if group != Some(offer.group) {
            group = Some(offer.group);
            ui.add_space(6.0);
            ui.label(egui::RichText::new(offer.group.label()).small().color(weak));
            ui.add_space(2.0);
        }
        let row = egui::Button::new(icons::with_label(offer.icon, Some(weak), &offer.name))
            .min_size(egui::vec2(ui.available_width(), 0.0));
        if ui.add(row).clicked() {
            chosen = Some(offer.id.clone());
        }
    }
    chosen
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
    let mut dropped_mesh = None;
    let mut picked_material = None;
    let mut dropped_material = None;
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
                    picked_mesh = mesh_picker(ui, world, entity, previews, &mut dropped_mesh);
                });
            }
            if rows.shows("Material") && world.has::<MaterialHandle>(entity) {
                widgets::property(ui, "Material", |ui| {
                    picked_material =
                        material_picker(ui, world, entity, previews, &mut dropped_material);
                });
            }
        },
    );
    // Posted as the asset's id, the form a scene file holds them in.
    let assets = world.resource::<Assets>();
    if let Some(id) = picked_mesh.and_then(|handle| assets.mesh_id(handle)) {
        assign_mesh(state, entity, id);
    }
    if let Some(id) = picked_material.and_then(|handle| assets.material_id(handle)) {
        assign_material(state, entity, id);
    }
    if let Some(id) = dropped_mesh {
        assign_mesh(state, entity, id);
    }
    if let Some(id) = dropped_material {
        assign_material(state, entity, id);
    }
}

/// Point `entity`'s mesh at asset `id` as an edit like any other, so it can be
/// undone, and an asset that is not a mesh is refused by name rather than
/// drawn as one.
fn assign_mesh(state: &mut EditorState, entity: Entity, id: AssetId) {
    let value = orrin_registry::Value::Asset(id);
    state.edit(entity, crate::scene::registry::MESH, value);
}

/// [`assign_mesh`] for the Material slot: a material asset, or an engine
/// material by its derived id.
fn assign_material(state: &mut EditorState, entity: Entity, id: AssetId) {
    let value = orrin_registry::Value::Asset(id);
    state.edit(entity, crate::scene::registry::MATERIAL, value);
}

/// Take an asset dropped on `slot`, if one was.
fn take_drop(slot: &egui::Response, dropped: &mut Option<AssetId>) {
    if !egui::DragAndDrop::has_payload_of_type::<AssetId>(&slot.ctx) {
        return;
    }
    if let Some(id) = slot.dnd_release_payload::<AssetId>() {
        *dropped = Some(*id);
    }
}

/// The file name of the asset a database-issued handle draws, for a slot that
/// names it.
fn database_name(world: &World, id: Option<AssetId>) -> Option<String> {
    let db = world.get_resource::<AssetDatabase>()?;
    db.get(id?).map(|entry| entry.name().to_owned())
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
    fallback: Option<String>,
    slot: impl FnOnce(&mut egui::Ui, &str) -> egui::Response,
    thumbnail: &mut dyn FnMut(&egui::Context, H) -> Option<egui::TextureHandle>,
) -> Option<H> {
    let name = current
        .and_then(|h| choices.iter().find(|choice| choice.handle == h))
        .map(|choice| choice.name.clone())
        .or(fallback)
        .unwrap_or_else(|| "—".to_owned());
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
    dropped: &mut Option<AssetId>,
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
    let imported = current.and_then(|handle| {
        let id = world.get_resource::<AssetDatabase>()?.mesh_id(handle);
        database_name(world, id)
    });

    picker(
        ui,
        "mesh",
        choices,
        current,
        imported,
        |ui, name| {
            let slot = widgets::asset_slot(ui, texture.as_ref(), icons::box_(), name, &subtitle);
            take_drop(&slot, dropped);
            slot.on_hover_text("Click to pick an engine mesh, or drop a mesh asset here")
        },
        &mut |ctx, handle| previews.mesh(ctx, world, handle),
    )
}

fn material_picker(
    ui: &mut egui::Ui,
    world: &World,
    entity: Entity,
    previews: &mut Previews,
    dropped: &mut Option<AssetId>,
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
    let imported = current.and_then(|handle| {
        let id = world.get_resource::<AssetDatabase>()?.material_id(handle);
        database_name(world, id)
    });

    picker(
        ui,
        "material",
        choices,
        current,
        imported,
        |ui, name| {
            let slot = widgets::asset_slot(ui, texture.as_ref(), icons::palette(), name, &subtitle);
            if world.get_resource::<AssetDatabase>().is_none() {
                return slot;
            }
            take_drop(&slot, dropped);
            slot.on_hover_text("Click to pick an engine material, or drop a material asset here")
        },
        &mut |ctx, handle| previews.material(ctx, world, handle),
    )
}

/// The texture slots a material asset's section offers, by row label and by
/// the name its file gives the slot.
const MAPS: [(&str, &str); 4] = [
    ("Albedo map", "albedo"),
    ("Normal map", "normal"),
    ("Metal/rough map", "metallic_roughness"),
    ("Emission map", "emissive"),
];

/// The selected entity's material, edited in place.
///
/// A material is shared by every entity that draws with it, so an edit here
/// changes all of them — which is what editing a material means. A material
/// asset is edited through the edit pipeline and saved to its `.omat` as each
/// gesture ends, so it can be undone with the scene's edits; its texture slots
/// take texture assets dropped on them. An engine or model material has no
/// file: its edit goes to the world-side copy and is queued for the renderer
/// (see [`AssetPreviews`]), for the session only. Neither is ever written into
/// a component.
fn material_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    previews: &mut Previews,
    filter: &Filter,
) {
    let Some(handle) = world.get::<MaterialHandle>(entity).map(|h| *h) else {
        return;
    };
    let asset = world.get_resource::<AssetDatabase>().and_then(|db| {
        let id = db.material_id(handle)?;
        Some((id, db.material_file(id)?.clone()))
    });
    let (before, textured) = match asset {
        Some((_, ref file)) => (file.material, file.textures.albedo.is_some()),
        None => {
            let Some(kept) = world
                .get_resource::<AssetPreviews>()
                .and_then(|kept| kept.material(handle).map(|p| (p.material, p.textured)))
            else {
                return;
            };
            kept
        }
    };
    let mut labels = vec![
        "Preview",
        "Base colour",
        "Metallic",
        "Roughness",
        "Emission",
        "Blend",
    ];
    if asset.is_some() {
        labels.extend(MAPS.map(|(label, _)| label));
    }
    let Some(rows) = filter.section("Material", &labels) else {
        return;
    };

    let mut after = before;
    let mut slots = asset.as_ref().map(|(_, file)| file.clone());
    let id = ui.id().with("material");
    let weak = ui.visuals().weak_text_color();
    let mut previewed = None;
    widgets::section(ui, id, icons::palette(), weak, "Material", None, |ui| {
        if rows.shows("Preview") {
            // Drawn by the renderer this frame, through the frame's own
            // shading; see `gfx::vulkan::material_preview`.
            let side = (ui.available_width() * 0.6).clamp(96.0, 160.0);
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                previewed = Some(widgets::material_preview(
                    ui,
                    material_preview::Viewer::Inspector,
                    handle,
                    side,
                ));
            });
            ui.add_space(8.0);
        }
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
        if let Some(file) = &mut slots {
            for (label, slot) in MAPS {
                if rows.shows(label) {
                    let texture = file.texture_mut(slot).expect("every row names a slot");
                    widgets::property(ui, label, |ui| texture_slot(ui, world, texture));
                }
            }
        }
    });

    if previewed.is_some() {
        state.material_previews[material_preview::Viewer::Inspector as usize] = previewed;
    }

    let edited = before.base_color != after.base_color
        || before.metallic != after.metallic
        || before.roughness != after.roughness
        || before.emissive != after.emissive;
    match (asset, slots) {
        (Some((id, file)), Some(mut changed)) => {
            if edited || changed.textures != file.textures {
                changed.material = after;
                state.edit_material(id, changed);
            }
        }
        _ => {
            if edited && let Some(mut kept) = world.get_resource_mut::<AssetPreviews>() {
                kept.edit_material(handle, |material| *material = after);
            }
        }
    }
}

/// One texture slot of a material asset: the texture's name, a drop target
/// for another, and a menu to empty it. Whether a drop is a texture is the
/// edit's to decide, so a refusal is named in the console like any other.
pub(super) fn texture_slot(ui: &mut egui::Ui, world: &World, texture: &mut Option<AssetId>) {
    let name = match *texture {
        Some(id) => database_name(world, Some(id)).unwrap_or_else(|| format!("missing {id}")),
        None => "—".to_owned(),
    };
    let slot =
        ui.add(egui::Button::new(figures(name)).min_size(egui::vec2(ui.available_width(), 0.0)));
    let mut dropped = None;
    take_drop(&slot, &mut dropped);
    if dropped.is_some() {
        *texture = dropped;
    }
    slot.on_hover_text("Drop a texture asset here. Right-click to empty the slot")
        .context_menu(|ui| {
            if ui
                .add_enabled(texture.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                *texture = None;
                ui.close_menu();
            }
        });
}

/// One section per Behaviour on the entity, in attach order: its lifecycle
/// rows, and, when its type is a registered component, its fields and a way to
/// remove it alone.
///
/// A Behaviour with no `[Component]` has no id for the edit pipeline to name,
/// so it cannot be removed here: a removal that skipped the pipeline would be
/// the one edit undo never saw.
#[cfg(feature = "scripting")]
fn script_sections(
    ui: &mut egui::Ui,
    world: &World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
    overridden: &[orrin_registry::ComponentId],
) {
    struct Shown {
        type_name: String,
        enabled: bool,
        faulted: bool,
        active: bool,
        started: bool,
    }
    // Copied out, because reading a Behaviour's fields through the registry
    // borrows the same storage.
    let shown: Vec<Shown> = world
        .get::<Scripts>(entity)
        .map(|scripts| {
            scripts
                .iter()
                .map(|script| Shown {
                    type_name: script.type_name.clone(),
                    enabled: script.enabled,
                    faulted: script.faulted,
                    active: script.active,
                    started: script.started,
                })
                .collect()
        })
        .unwrap_or_default();

    for script in shown {
        let vtable = registry.components().find(|vtable| {
            vtable
                .script
                .as_ref()
                .is_some_and(|binding| binding.type_name == script.type_name)
        });
        let title = vtable.map_or_else(
            || {
                let qualified = script.type_name.split(',').next().unwrap_or_default();
                qualified
                    .rsplit(['.', '+'])
                    .next()
                    .unwrap_or(qualified)
                    .to_owned()
            },
            |vtable| vtable.name.to_string(),
        );
        let before = vtable.and_then(|vtable| vtable.read(world, entity));
        let annotations = vtable
            .and_then(|vtable| vtable.script.as_ref())
            .map_or(&[][..], |binding| &binding.fields[..]);
        let hidden = |name: &str| {
            annotations
                .iter()
                .any(|info| info.hidden && info.name == name)
        };
        let mut labels = vec!["Behaviour", "Enabled", "State"];
        if let Some(
            orrin_registry::Value::Struct(fields) | orrin_registry::Value::Enum { fields, .. },
        ) = &before
        {
            labels.extend(
                fields
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .filter(|name| !hidden(name)),
            );
        }
        let Some(rows) = filter.section(&title, &labels) else {
            continue;
        };

        let mut enabled = script.enabled;
        let mut clear_fault = false;
        let mut after = before.clone();
        let mut remove = false;
        let mut menu = |ui: &mut egui::Ui| {
            if ui.button("Remove component").clicked() {
                remove = true;
                ui.close_menu();
            }
        };
        let id = ui.id().with(("script", script.type_name.as_str()));
        let title_colour = vtable
            .is_some_and(|vtable| overridden.contains(&vtable.id))
            .then_some(crate::editor::theme::PREFAB);
        let touched = widgets::section_titled(
            ui,
            id,
            icons::file_code(),
            crate::editor::theme::KIND_SCRIPT,
            &title,
            title_colour,
            vtable.map(|_| &mut menu as &mut dyn FnMut(&mut egui::Ui)),
            |ui| {
                if rows.shows("Behaviour") {
                    widgets::property(ui, "Behaviour", |ui| {
                        ui.label(figures(script.type_name.clone()))
                    });
                }
                if rows.shows("Enabled") {
                    // Only the desired state is written here; the script tick
                    // sees the change next frame and dispatches
                    // OnEnable/OnDisable itself.
                    widgets::property(ui, "Enabled", |ui| ui.checkbox(&mut enabled, ""));
                }
                if rows.shows("State") {
                    widgets::property(ui, "State", |ui| {
                        if script.faulted {
                            // A hook threw and the engine disabled the script
                            // (see the log for the exception). Clearing re-arms
                            // it from next tick — the manual counterpart to
                            // "re-enable on hot reload".
                            icons::labelled(
                                ui,
                                icons::warning(),
                                crate::editor::theme::ERROR,
                                "faulted",
                            );
                            if ui.small_button("Clear fault").clicked() {
                                clear_fault = true;
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
                match after.as_mut() {
                    Some(after) => script_fields(ui, world, after, annotations),
                    None => false,
                }
            },
        )
        .unwrap_or(false);
        drop(menu);

        if (enabled != script.enabled || clear_fault)
            && let Some(mut scripts) = world.get_mut::<Scripts>(entity)
            && let Some(live) = scripts.get_mut(&script.type_name)
        {
            live.enabled = enabled;
            if clear_fault {
                live.faulted = false;
            }
        }
        if let Some(vtable) = vtable {
            if remove {
                state.request_remove_component(entity, vtable.id.clone());
            } else if touched && let Some(after) = after {
                state.edit(entity, vtable.id.clone(), after);
            }
        }
    }
}

/// A Behaviour's fields, as their attributes ask, with every entity reference
/// drawn as an [`entity_slot`].
#[cfg(feature = "scripting")]
fn script_fields(
    ui: &mut egui::Ui,
    world: &World,
    value: &mut orrin_registry::Value,
    fields: &[orrin_registry::FieldInfo],
) -> bool {
    orrin_registry::inspect::annotated(ui, value, fields, &mut |ui, id| entity_slot(ui, world, id))
}

/// One entity reference: the target's name, a drop target for a row dragged
/// out of the hierarchy, and a menu to clear it — [`texture_slot`]'s
/// interaction, over the world rather than the asset index.
///
/// A drag carrying several rows is refused rather than resolved to one of
/// them: which one was meant is not something the slot can know.
#[cfg(feature = "scripting")]
fn entity_slot(ui: &mut egui::Ui, world: &World, id: &mut orrin_registry::EntityId) -> bool {
    use crate::scene::entity_ref;

    let name = match entity_ref::entity_of(world, *id) {
        Some(target) => world.get::<Name>(target).map_or_else(
            || format!("Entity {}", target.index()),
            |name| name.0.clone(),
        ),
        None if id.is_nil() => "None".to_owned(),
        None => "Missing entity".to_owned(),
    };
    let slot =
        ui.add(egui::Button::new(figures(name)).min_size(egui::vec2(ui.available_width(), 0.0)));
    let mut changed = false;
    if egui::DragAndDrop::has_payload_of_type::<Vec<Entity>>(&slot.ctx)
        && let Some(dragged) = slot.dnd_release_payload::<Vec<Entity>>()
        && let [target] = dragged[..]
    {
        let dropped = entity_ref::id_of(world, target);
        changed = dropped != *id;
        *id = dropped;
    }
    slot.on_hover_text("Drop an entity from the hierarchy here. Right-click to clear")
        .context_menu(|ui| {
            if ui
                .add_enabled(!id.is_nil(), egui::Button::new("Clear"))
                .clicked()
            {
                *id = orrin_registry::EntityId::NIL;
                changed = true;
                ui.close_menu();
            }
        });
    changed
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

/// The scene's physics, with its layer matrix as Unity draws one: a triangle
/// of checkboxes, each pair once, rather than the eight bitmasks it is stored
/// as.
fn physics_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    use crate::scene::{PHYSICS_LAYERS, PhysicsSettings};

    let Some(before) = world.get::<PhysicsSettings>(entity).map(|s| s.clone()) else {
        return;
    };
    let labels = [
        "Gravity",
        "Steps per second",
        "Max steps per frame",
        "Layers",
    ];
    let Some(rows) = filter.section("Physics Settings", &labels) else {
        return;
    };
    let mut settings = before.clone();
    let defaults = PhysicsSettings::default();
    widgets::section(
        ui,
        ui.id().with("physics_settings"),
        icons::globe(),
        crate::editor::theme::KIND_PHYSICS,
        "Physics Settings",
        None,
        |ui| {
            if rows.shows("Gravity") {
                let mut gravity = settings.gravity.to_array();
                transform_row(
                    ui,
                    "Gravity",
                    &mut gravity,
                    defaults.gravity.to_array(),
                    0.1,
                    " m/s²",
                );
                settings.gravity = glam::Vec3::from_array(gravity);
            }
            if rows.shows("Steps per second") {
                widgets::property(ui, "Steps per second", |ui| {
                    ui.add(
                        egui::Slider::new(&mut settings.steps_per_second, 10..=240).suffix(" Hz"),
                    )
                    .on_hover_text(
                        "Higher stacks stiffer and costs proportionally more; \
                             the frame interpolates between steps either way",
                    );
                });
            }
            if rows.shows("Max steps per frame") {
                widgets::property(ui, "Max steps per frame", |ui| {
                    ui.add(egui::Slider::new(&mut settings.max_steps_per_frame, 1..=16))
                        .on_hover_text(
                            "A slower frame drops the rest of its time rather than \
                             spend more steps catching up",
                        );
                });
            }
            if rows.shows("Layers") {
                widgets::property(ui, "Layers", |ui| {
                    egui::Grid::new(ui.id().with("layer_matrix"))
                        .spacing([2.0, 2.0])
                        .show(ui, |ui| {
                            ui.label("");
                            for column in 0..PHYSICS_LAYERS {
                                ui.weak(column.to_string());
                            }
                            ui.end_row();
                            for row in 0..PHYSICS_LAYERS {
                                ui.weak(row.to_string());
                                for column in 0..=row {
                                    let mut collide = settings.collides(row, column);
                                    let hint = format!("Layer {row} meets layer {column}");
                                    if ui.checkbox(&mut collide, "").on_hover_text(hint).changed() {
                                        settings.set_collides(row, column, collide);
                                    }
                                }
                                ui.end_row();
                            }
                        });
                });
            }
        },
    );
    let after = settings.to_value();
    if !orrin_registry::diff(&before.to_value(), &after).is_empty() {
        state.edit(entity, crate::scene::registry::PHYSICS_SETTINGS, after);
    }
}

/// Drawn by hand for its clip, which the generic row could only show as an id
/// and could not check: an [`AssetId`] does not say what kind of asset it
/// names, so only a section that can see the database can refuse a texture.
fn audio_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
    overridden: &[orrin_registry::ComponentId],
) {
    use crate::scene::registry::AUDIO_SOURCE;
    use crate::scene::{AudioSource, Rolloff};

    let Some(before) = world.get::<AudioSource>(entity).map(|s| *s) else {
        return;
    };
    let labels = [
        "Clip",
        "Volume",
        "Loop",
        "Play on start",
        "Spatial",
        "Min distance",
        "Max distance",
        "Rolloff",
    ];
    let Some(rows) = filter.section("Audio Source", &labels) else {
        return;
    };
    let mut source = before;
    let mut remove = false;
    let mut menu = |ui: &mut egui::Ui| {
        if ui.button("Remove component").clicked() {
            remove = true;
            ui.close_menu();
        }
    };
    let title = overridden
        .contains(&AUDIO_SOURCE)
        .then_some(crate::editor::theme::PREFAB);
    widgets::section_titled(
        ui,
        ui.id().with("audio_source"),
        icons::speaker_high(),
        crate::editor::theme::KIND_AUDIO,
        "Audio Source",
        title,
        Some(&mut menu),
        |ui| {
            if rows.shows("Clip") {
                clip_row(ui, world, &mut source.clip);
            }
            if rows.shows("Volume") {
                widgets::property(ui, "Volume", |ui| {
                    let slider = egui::Slider::new(&mut source.volume, 0.0..=1.0)
                        .clamping(egui::SliderClamping::Never);
                    ui.add(slider)
                        .on_hover_text("A linear gain, not decibels: 0.5 is half the amplitude");
                });
                source.volume = source.volume.max(0.0);
            }
            if rows.shows("Loop") {
                widgets::property(ui, "Loop", |ui| ui.checkbox(&mut source.looping, ""));
            }
            if rows.shows("Play on start") {
                widgets::property(ui, "Play on start", |ui| {
                    ui.checkbox(&mut source.play_on_start, "")
                        .on_hover_text("Starts when the game does, or when it appears in one")
                });
            }
            if rows.shows("Spatial") {
                widgets::property(ui, "Spatial", |ui| {
                    ui.checkbox(&mut source.spatial, "")
                        .on_hover_text("Heard from where it stands, rather than in both ears alike")
                });
            }
            let spatial = source.spatial;
            for (label, distance) in [
                ("Min distance", &mut source.min_distance),
                ("Max distance", &mut source.max_distance),
            ] {
                if rows.shows(label) {
                    widgets::property(ui, label, |ui| {
                        let field = egui::DragValue::new(distance)
                            .speed(0.1)
                            .range(0.0..=f32::MAX)
                            .suffix(" m");
                        ui.add_enabled(spatial, field)
                    });
                }
            }
            if rows.shows("Rolloff") {
                widgets::property(ui, "Rolloff", |ui| {
                    ui.add_enabled_ui(spatial, |ui| {
                        egui::ComboBox::from_id_salt(ui.id().with("rolloff"))
                            .width(ui.available_width())
                            .selected_text(format!("{:?}", source.rolloff))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut source.rolloff,
                                    Rolloff::Inverse,
                                    "Inverse",
                                )
                                .on_hover_text("Halves with every doubling of distance");
                                ui.selectable_value(&mut source.rolloff, Rolloff::Linear, "Linear")
                                    .on_hover_text("Falls to silence at the max distance");
                            });
                    });
                });
            }
        },
    );
    if remove {
        state.request_remove_component(entity, AUDIO_SOURCE);
    } else if source != before {
        state.edit(entity, AUDIO_SOURCE, source.to_value());
    }
}

/// Whether `id` is a sound in the project, and if not, why not by name.
fn is_sound(world: &World, id: AssetId) -> Result<(), String> {
    match world.get_resource::<AssetDatabase>() {
        Some(db) => db.audio_file(id).map(drop),
        None => Err("there is no project to find a sound in".to_owned()),
    }
}

/// The clip: a slot naming the sound, which opens a list of the project's
/// sounds and takes a dropped one, and beside it a button that auditions it.
/// Anything dropped that is not a sound is refused by name.
fn clip_row(ui: &mut egui::Ui, world: &World, clip: &mut AssetId) {
    let current = (!clip.is_nil()).then_some(*clip);
    let (choices, title, subtitle, about) = {
        let db = world.get_resource::<AssetDatabase>();
        let mut choices: Vec<Choice<AssetId>> = db
            .iter()
            .flat_map(|db| db.iter())
            .filter(|entry| entry.kind == crate::asset_db::AssetKind::Audio)
            .map(|entry| Choice {
                name: entry.path.clone(),
                handle: entry.id,
            })
            .collect();
        choices.sort_by(|a, b| a.name.cmp(&b.name));
        // A missing sound keeps its id on hover: it is the only clue to what
        // the scene was pointing at.
        let (title, subtitle, about) = match current {
            None => ("No sound".to_owned(), String::new(), String::new()),
            Some(id) => match db.as_ref().and_then(|db| db.get(id)) {
                Some(entry) => {
                    let (folder, file) = entry.path.rsplit_once('/').unwrap_or(("", &entry.path));
                    let about = format!("{}\n", entry.path);
                    (file.to_owned(), folder.to_owned(), about)
                }
                None => (
                    "Missing sound".to_owned(),
                    "Not in the project".to_owned(),
                    format!("{id}\n"),
                ),
            },
        };
        (choices, title, subtitle, about)
    };
    let previewing = world
        .get_resource::<crate::audio::Audio>()
        .and_then(|audio| audio.previewing());

    let mut dropped = None;
    let mut cleared = false;
    let mut toggled = false;
    let picked = widgets::property_with(
        ui,
        "Clip",
        |ui| {
            picker(
                ui,
                "clip",
                choices,
                current,
                None,
                |ui, _| {
                    let slot =
                        widgets::asset_slot(ui, None, icons::file_audio(), &title, &subtitle);
                    if slot.contains_pointer()
                        && let Some(payload) = egui::DragAndDrop::payload::<AssetId>(ui.ctx())
                        && is_sound(world, *payload).is_err()
                    {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::NoDrop);
                    }
                    take_drop(&slot, &mut dropped);
                    let hint = format!("{about}Click to pick a sound, or drop one here");
                    let slot = slot.on_hover_text(hint);
                    slot.context_menu(|ui| {
                        if ui
                            .add_enabled(current.is_some(), egui::Button::new("Clear"))
                            .clicked()
                        {
                            cleared = true;
                            ui.close_menu();
                        }
                    });
                    slot
                },
                &mut |_, _| None,
            )
        },
        |ui| {
            let Some(id) = current else {
                return;
            };
            let playing = previewing == Some(id);
            let (icon, hint) = if playing {
                (icons::stop(), "Stop the preview")
            } else {
                (icons::play(), "Preview the sound")
            };
            // Drawn rather than an `egui::Button`, which centres the glyph's
            // line box: the icon font sits its glyphs low for text, so the
            // square came out below the middle of its frame.
            let side = ui.available_width().min(ui.available_height());
            let (rect, response) =
                ui.allocate_exact_size(egui::Vec2::splat(side), egui::Sense::click());
            let visuals = ui.style().interact(&response);
            if playing || response.hovered() {
                ui.painter().rect(
                    rect,
                    visuals.corner_radius,
                    visuals.weak_bg_fill,
                    visuals.bg_stroke,
                    egui::StrokeKind::Inside,
                );
            }
            let weak = ui.visuals().weak_text_color();
            icons::paint_ink_centred(ui.painter(), icon, 12.0, rect.center(), weak);
            toggled = response.on_hover_text(hint).clicked();
        },
    );

    if toggled
        && let Some(id) = current
        && let Some(mut audio) = world.get_resource_mut::<crate::audio::Audio>()
    {
        if previewing == Some(id) {
            audio.stop_preview();
        } else {
            audio.preview(id);
        }
    }
    if let Some(id) = dropped {
        match is_sound(world, id) {
            Ok(()) => *clip = id,
            Err(error) => tracing::warn!("Audio Source clip: {error}"),
        }
    }
    if let Some(id) = picked {
        *clip = id;
    }
    if cleared {
        *clip = AssetId::NIL;
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
            [
                "orrin.tag",
                "orrin.collider",
                "orrin.rigid_body",
                "orrin.audio_listener",
                "orrin.spin",
                "orrin.locked",
                "orrin.render_settings",
                "orrin.camera"
            ],
            "registration order changed, or a component gained/lost a section"
        );
    }

    /// A dropped asset goes through the registry like any other edit, so an
    /// asset that is not a mesh is refused by its path and the mesh stays.
    /// A dropped asset is an edit like any other: refused by name when it is
    /// not a mesh, and undone in one step when it is.
    #[test]
    fn a_dropped_mesh_asset_is_an_undoable_edit_and_anything_else_is_refused_by_name() {
        let registry = registry();
        let dir = crate::asset_db::tests::Dir::new("inspector-drop");
        dir.write("rock.png", b"");
        dir.write("tree.glb", b"");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let rock = db.find("rock.png").unwrap().id;
        let tree = db.find("tree.glb").unwrap().id;

        let mut world = World::new();
        world.insert_resource(db);
        world.insert_resource(LogBuffer::default());
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(0));
        world.insert_resource(assets);
        let entity = world.spawn();
        world.insert(entity, MeshHandle(0));
        world.insert(entity, Name::new("Crate"));
        let mut state = EditorState::default();

        assign_mesh(&mut state, entity, rock);
        state.apply(&mut world, &registry);
        assert_eq!(*world.get::<MeshHandle>(entity).unwrap(), MeshHandle(0));
        let logged: Vec<String> = world
            .resource::<LogBuffer>()
            .iter()
            .map(|entry| entry.message.clone())
            .collect();
        assert_eq!(
            logged,
            ["orrin.mesh on Crate: expected a mesh asset, found `rock.png`, which is a texture"]
        );
        assert_eq!(
            state.history.undo_label(),
            None,
            "a refused edit is no step"
        );

        assign_mesh(&mut state, entity, tree);
        state.apply(&mut world, &registry);
        let handle = *world.get::<MeshHandle>(entity).unwrap();
        assert_eq!(
            world.resource::<AssetDatabase>().mesh_id(handle),
            Some(tree)
        );

        state.request_undo();
        state.apply(&mut world, &registry);
        assert_eq!(*world.get::<MeshHandle>(entity).unwrap(), MeshHandle(0));
    }

    /// A material asset dropped on the mesh renderer is the same kind of edit
    /// a dropped mesh is.
    #[test]
    fn a_dropped_material_asset_is_an_undoable_edit_and_anything_else_is_refused_by_name() {
        let registry = registry();
        let dir = crate::asset_db::tests::Dir::new("inspector-drop-material");
        dir.write("rock.png", b"");
        dir.write(
            "brick.omat",
            crate::asset_db::material::write(&Default::default()).as_bytes(),
        );
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let rock = db.find("rock.png").unwrap().id;
        let brick = db.find("brick.omat").unwrap().id;

        let mut world = World::new();
        world.insert_resource(db);
        world.insert_resource(LogBuffer::default());
        let mut assets = Assets::new();
        assets.insert_material("clay", MaterialHandle(0));
        world.insert_resource(assets);
        let entity = world.spawn();
        world.insert(entity, MeshHandle(0));
        world.insert(entity, MaterialHandle(0));
        world.insert(entity, Name::new("Crate"));
        let mut state = EditorState::default();

        assign_material(&mut state, entity, rock);
        state.apply(&mut world, &registry);
        assert_eq!(
            *world.get::<MaterialHandle>(entity).unwrap(),
            MaterialHandle(0)
        );
        let logged: Vec<String> = world
            .resource::<LogBuffer>()
            .iter()
            .map(|entry| entry.message.clone())
            .collect();
        assert_eq!(
            logged,
            [
                "orrin.material on Crate: expected a material asset, found `rock.png`, which is a \
                 texture"
            ]
        );

        assign_material(&mut state, entity, brick);
        state.apply(&mut world, &registry);
        let handle = *world.get::<MaterialHandle>(entity).unwrap();
        assert_eq!(
            world.resource::<AssetDatabase>().material_id(handle),
            Some(brick)
        );
        assert_eq!(state.history.undo_label(), Some("Edit Material"));

        state.request_undo();
        state.apply(&mut world, &registry);
        assert_eq!(
            *world.get::<MaterialHandle>(entity).unwrap(),
            MaterialHandle(0)
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
                ..Collider::default()
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

    /// Every text the frame painted, with where.
    #[cfg(feature = "scripting")]
    fn texts(output: &egui::FullOutput) -> Vec<(String, egui::Rect)> {
        fn collect(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
            match shape {
                egui::Shape::Text(text) => {
                    out.push((text.galley.text().to_owned(), text.visual_bounding_rect()))
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut out);
        }
        out
    }

    /// A Behaviour's entity field reads as the entity it names, as `None` when
    /// it names nothing, and as missing when the entity it named is gone.
    #[cfg(feature = "scripting")]
    #[test]
    fn an_entity_field_shows_its_targets_name() {
        let mut world = World::new();
        let door = world.spawn();
        world.insert(door, Name::new("Door"));
        let gone = world.spawn();
        let gone_id = crate::scene::entity_ref::id_of(&world, gone);
        world.despawn(gone);
        let mut bag = Value::strukt([
            (
                "Target",
                Value::Entity(crate::scene::entity_ref::id_of(&world, door)),
            ),
            ("Nothing", Value::Entity(orrin_registry::EntityId::NIL)),
            ("Lost", Value::Entity(gone_id)),
        ]);
        let ctx = egui::Context::default();
        let mut painted = Vec::new();
        for _ in 0..2 {
            let output = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    script_fields(ui, &world, &mut bag, &[]);
                });
            });
            painted = texts(&output);
        }
        let shown: Vec<&str> = painted.iter().map(|(text, _)| text.as_str()).collect();
        for expected in ["Door", "None", "Missing entity"] {
            assert!(shown.contains(&expected), "{expected} not in {shown:?}");
        }
    }

    /// A row dragged out of the hierarchy and let go over the slot points the
    /// field at it.
    #[cfg(feature = "scripting")]
    #[test]
    fn dropping_a_hierarchy_row_on_an_entity_field_assigns_it() {
        let mut world = World::new();
        let door = world.spawn();
        world.insert(door, Name::new("Door"));
        let mut bag = Value::strukt([("Target", Value::Entity(orrin_registry::EntityId::NIL))]);
        let ctx = egui::Context::default();
        let draw = |input: egui::RawInput, bag: &mut Value| {
            let mut changed = false;
            let output = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    changed |= script_fields(ui, &world, bag, &[]);
                });
            });
            (changed, texts(&output))
        };
        draw(Default::default(), &mut bag);
        let (_, painted) = draw(Default::default(), &mut bag);
        let slot = painted.iter().find(|(text, _)| text == "None").unwrap().1;

        egui::DragAndDrop::set_payload(&ctx, vec![door]);
        let release = egui::RawInput {
            events: vec![
                egui::Event::PointerMoved(slot.center()),
                egui::Event::PointerButton {
                    pos: slot.center(),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ],
            ..Default::default()
        };
        let (changed, _) = draw(release, &mut bag);
        assert!(changed);
        assert_eq!(
            bag.field("Target"),
            Some(&Value::Entity(crate::scene::entity_ref::id_of(
                &world, door
            )))
        );
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
