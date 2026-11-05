//! The inspector's sections for the retained UI's components.
//!
//! Each edits a detached copy and hands the edit pipeline the result, as every
//! other section does. Colours are sRGB with straight alpha, so they go through
//! egui's sRGB picker as bytes, and a colour that was not touched keeps the
//! floats it had rather than being rounded to the picker's 1/255 steps.

use glam::Vec2;
use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, ComponentId, Reflect};

use super::inspector::{Filter, texture_slot};
use crate::editor::icons;
use crate::editor::state::EditorState;
use crate::editor::widgets;
use crate::scene::TextAnchor;
use crate::scene::registry::{BUTTON, CANVAS, IMAGE, RECT_TRANSFORM, TEXT};
use crate::ui::components::AnchorPreset;
use crate::ui::{Button, Canvas, Image, RectTransform, ScaleMode, Text, UiTree};

/// Every UI section the entity has a component for.
pub fn sections(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    canvas_section(ui, world, state, entity, filter);
    rect_section(ui, world, state, entity, filter);
    image_section(ui, world, state, entity, filter);
    text_section(ui, world, state, entity, filter);
    button_section(ui, world, state, entity, filter);
}

/// Post `after` as an edit of `id` if it differs from `before`.
fn edit<T: Reflect>(
    state: &mut EditorState,
    entity: Entity,
    id: ComponentId,
    before: &T,
    after: &T,
) {
    let after = after.to_value();
    if !orrin_registry::diff(&before.to_value(), &after).is_empty() {
        state.edit(entity, id, after);
    }
}

/// A section titled `title`, its remove menu included, unless the filter hides
/// it. Returns whether the remove was asked for.
fn section(
    ui: &mut egui::Ui,
    state: &mut EditorState,
    (entity, id): (Entity, ComponentId),
    (title, icon): (&str, icons::Icon),
    labels: &[&str],
    filter: &Filter,
    body: impl FnOnce(&mut egui::Ui),
) {
    if filter.section(title, labels).is_none() {
        return;
    }
    let mut remove = false;
    let mut menu = |ui: &mut egui::Ui| {
        if ui.button("Remove component").clicked() {
            remove = true;
            ui.close_menu();
        }
    };
    let tint = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with(("ui component", id.as_str())),
        icon,
        tint,
        title,
        Some(&mut menu),
        body,
    );
    if remove {
        state.request_remove_component(entity, id);
    }
}

fn pair(ui: &mut egui::Ui, label: &str, value: &mut Vec2, speed: f64) {
    widgets::property(ui, label, |ui| {
        ui.horizontal(|ui| {
            let width = ((ui.available_width() - 4.0) / 2.0).max(40.0);
            for (axis, v) in [("X", &mut value.x), ("Y", &mut value.y)] {
                ui.add_sized(
                    [width, ui.spacing().interact_size.y],
                    egui::DragValue::new(v)
                        .speed(speed)
                        .prefix(format!("{axis} ")),
                );
            }
        });
    });
}

fn colour(ui: &mut egui::Ui, label: &str, rgba: &mut [f32; 4]) {
    widgets::property(ui, label, |ui| {
        let before = crate::scene::color_bytes(*rgba);
        let mut bytes = before;
        ui.color_edit_button_srgba_unmultiplied(&mut bytes);
        if bytes != before {
            *rgba = bytes.map(|channel| f32::from(channel) / 255.0);
        }
    });
}

fn raycast(ui: &mut egui::Ui, value: &mut bool) {
    widgets::property(ui, "Raycast target", |ui| {
        ui.checkbox(value, "")
            .on_hover_text("Stops the pointer, so what is behind it gets no events");
    });
}

fn canvas_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<Canvas>(entity).map(|canvas| *canvas) else {
        return;
    };
    let mut canvas = before;
    let labels = ["Scale mode", "Reference", "Match", "Sort order"];
    section(
        ui,
        state,
        (entity, CANVAS),
        ("Canvas", icons::image()),
        &labels,
        filter,
        |ui| {
            widgets::property(ui, "Scale mode", |ui| {
                egui::ComboBox::from_id_salt("canvas scale mode")
                    .selected_text(match canvas.scale_mode {
                        ScaleMode::ConstantPixel => "Constant pixel size",
                        ScaleMode::ScaleWithScreen => "Scale with screen size",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut canvas.scale_mode,
                            ScaleMode::ConstantPixel,
                            "Constant pixel size",
                        );
                        ui.selectable_value(
                            &mut canvas.scale_mode,
                            ScaleMode::ScaleWithScreen,
                            "Scale with screen size",
                        );
                    });
            });
            if canvas.scale_mode == ScaleMode::ScaleWithScreen {
                pair(ui, "Reference", &mut canvas.reference_resolution, 1.0);
                widgets::property(ui, "Match", |ui| {
                    ui.add(egui::Slider::new(
                        &mut canvas.match_width_or_height,
                        0.0..=1.0,
                    ))
                    .on_hover_text("0 scales with the window's width, 1 with its height");
                });
            }
            widgets::property(ui, "Sort order", |ui| {
                ui.add(egui::DragValue::new(&mut canvas.sort_order))
                    .on_hover_text("Canvases draw lowest first");
            });
        },
    );
    edit(state, entity, CANVAS, &before, &canvas);
}

fn rect_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<RectTransform>(entity).map(|rect| *rect) else {
        return;
    };
    // What the anchors are fractions of, so a preset can keep the element in
    // place; the window, for an element the layout has not reached yet.
    let parent = world
        .get_resource::<UiTree>()
        .and_then(|tree| {
            let index = tree.layout.index_of(entity)?;
            Some(tree.layout.elements()[index].parent_size)
        })
        .unwrap_or(Vec2::new(1920.0, 1080.0));
    let mut rect = before;
    let labels = [
        "Anchors",
        "Position",
        "Size",
        "Anchor min",
        "Anchor max",
        "Pivot",
        "Scale",
    ];
    section(
        ui,
        state,
        (entity, RECT_TRANSFORM),
        ("Rect Transform", icons::move_()),
        &labels,
        filter,
        |ui| {
            widgets::property(ui, "Anchors", |ui| presets(ui, &mut rect, parent))
                .on_hover_text("Shift also sets the pivot, Alt also moves it to its anchors");
            pair(ui, "Position", &mut rect.anchored_position, 1.0);
            pair(ui, "Size", &mut rect.size_delta, 1.0);
            pair(ui, "Anchor min", &mut rect.anchor_min, 0.01);
            pair(ui, "Anchor max", &mut rect.anchor_max, 0.01);
            pair(ui, "Pivot", &mut rect.pivot, 0.01);
            pair(ui, "Scale", &mut rect.scale, 0.01);
        },
    );
    edit(state, entity, RECT_TRANSFORM, &before, &rect);
}

/// uGUI's anchor preset grid: a column per horizontal preset, a row per
/// vertical one, each drawn as where it puts a box in its parent.
fn presets(ui: &mut egui::Ui, rect: &mut RectTransform, parent: Vec2) -> egui::Response {
    let current = [
        AnchorPreset::of(rect.anchor_min.x, rect.anchor_max.x),
        AnchorPreset::of(rect.anchor_min.y, rect.anchor_max.y),
    ];
    const CELL: f32 = 18.0;
    let (area, response) = ui.allocate_exact_size(
        egui::vec2(CELL * 4.0 + 6.0, CELL * 4.0 + 6.0),
        egui::Sense::hover(),
    );
    let visuals = ui.visuals().clone();
    for (row, down) in AnchorPreset::ALL.into_iter().enumerate() {
        for (column, across) in AnchorPreset::ALL.into_iter().enumerate() {
            let cell = egui::Rect::from_min_size(
                area.min + egui::vec2(column as f32 * (CELL + 2.0), row as f32 * (CELL + 2.0)),
                egui::Vec2::splat(CELL),
            );
            let id = ui.id().with(("anchor preset", row, column));
            let hit = ui.interact(cell, id, egui::Sense::click());
            let on = current == [Some(across), Some(down)];
            let stroke = if on || hit.hovered() {
                visuals.selection.stroke
            } else {
                visuals.widgets.inactive.bg_stroke
            };
            let painter = ui.painter();
            painter.rect_stroke(cell, 1.0, stroke, egui::StrokeKind::Inside);
            let span = |preset: AnchorPreset, min: f32, max: f32| match preset {
                AnchorPreset::Min => (min + 2.0, min + 7.0),
                AnchorPreset::Middle => ((min + max) / 2.0 - 2.5, (min + max) / 2.0 + 2.5),
                AnchorPreset::Max => (max - 7.0, max - 2.0),
                AnchorPreset::Stretch => (min + 2.0, max - 2.0),
            };
            let (x0, x1) = span(across, cell.left(), cell.right());
            let (y0, y1) = span(down, cell.top(), cell.bottom());
            let fill = if on {
                visuals.selection.bg_fill
            } else {
                visuals.widgets.inactive.fg_stroke.color
            };
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
                0.0,
                fill,
            );
            if hit.clicked() {
                let modifiers = ui.input(|input| input.modifiers);
                rect.apply_preset(parent, [across, down], modifiers.shift, modifiers.alt);
            }
        }
    }
    response
}

fn image_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<Image>(entity).map(|image| *image) else {
        return;
    };
    let mut image = before;
    let labels = ["Texture", "Colour", "Slice", "Raycast target"];
    section(
        ui,
        state,
        (entity, IMAGE),
        ("Image", icons::image()),
        &labels,
        filter,
        |ui| {
            widgets::property(ui, "Texture", |ui| {
                texture_slot(ui, world, &mut image.texture)
            });
            colour(ui, "Colour", &mut image.color);
            widgets::property(ui, "Slice", |ui| {
                ui.horizontal(|ui| {
                    for (side, value) in ["L", "T", "R", "B"].into_iter().zip(&mut image.slice) {
                        ui.add(
                            egui::DragValue::new(value)
                                .speed(1.0)
                                .range(0.0..=f32::MAX)
                                .prefix(format!("{side} ")),
                        );
                    }
                })
                .response
                .on_hover_text(
                    "Nine-slice borders, in pixels of the texture; zero stretches it whole",
                );
            });
            raycast(ui, &mut image.raycast_target);
        },
    );
    edit(state, entity, IMAGE, &before, &image);
}

fn text_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<Text>(entity).map(|text| text.clone()) else {
        return;
    };
    let mut text = before.clone();
    let labels = [
        "Text",
        "Font",
        "Size",
        "Colour",
        "Alignment",
        "Wrap",
        "Raycast target",
    ];
    section(
        ui,
        state,
        (entity, TEXT),
        ("Text", icons::text_t()),
        &labels,
        filter,
        |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut text.text)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            widgets::property(ui, "Font", |ui| font_slot(ui, world, &mut text.font));
            widgets::property(ui, "Size", |ui| {
                ui.add(
                    egui::DragValue::new(&mut text.size)
                        .speed(0.25)
                        .range(1.0..=512.0),
                );
            });
            colour(ui, "Colour", &mut text.color);
            widgets::property(ui, "Alignment", |ui| {
                egui::ComboBox::from_id_salt("text alignment")
                    .selected_text(anchor_name(text.anchor))
                    .show_ui(ui, |ui| {
                        for anchor in (0..9).filter_map(TextAnchor::from_u32) {
                            ui.selectable_value(&mut text.anchor, anchor, anchor_name(anchor));
                        }
                    });
            });
            widgets::property(ui, "Wrap", |ui| {
                ui.checkbox(&mut text.wrap, "")
                    .on_hover_text("Off, a line runs past the sides rather than breaking");
            });
            raycast(ui, &mut text.raycast_target);
        },
    );
    edit(state, entity, TEXT, &before, &text);
}

fn anchor_name(anchor: TextAnchor) -> &'static str {
    const NAMES: [&str; 9] = [
        "Upper left",
        "Upper centre",
        "Upper right",
        "Middle left",
        "Middle centre",
        "Middle right",
        "Lower left",
        "Lower centre",
        "Lower right",
    ];
    NAMES[anchor as usize]
}

/// A font slot: the font's name, a drop target for another, and a menu to go
/// back to the engine's own face.
fn font_slot(ui: &mut egui::Ui, world: &World, font: &mut Option<AssetId>) {
    let name = match *font {
        Some(id) => world
            .get_resource::<crate::asset_db::AssetDatabase>()
            .and_then(|db| db.get(id).map(|entry| entry.name().to_owned()))
            .unwrap_or_else(|| format!("missing {id}")),
        None => "Built-in".to_owned(),
    };
    let slot = ui.add(
        egui::Button::new(super::figures(name)).min_size(egui::vec2(ui.available_width(), 0.0)),
    );
    if egui::DragAndDrop::has_payload_of_type::<AssetId>(&slot.ctx)
        && let Some(id) = slot.dnd_release_payload::<AssetId>()
    {
        *font = Some(*id);
    }
    slot.on_hover_text("Drop a .ttf or .otf asset here. Right-click for the built-in face")
        .context_menu(|ui| {
            if ui
                .add_enabled(font.is_some(), egui::Button::new("Built-in"))
                .clicked()
            {
                *font = None;
                ui.close_menu();
            }
        });
}

fn button_section(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    entity: Entity,
    filter: &Filter,
) {
    let Some(before) = world.get::<Button>(entity).map(|button| *button) else {
        return;
    };
    let mut button = before;
    let labels = ["Normal", "Hover", "Pressed", "Disabled", "Interactable"];
    section(
        ui,
        state,
        (entity, BUTTON),
        ("Button", icons::pointer()),
        &labels,
        filter,
        |ui| {
            colour(ui, "Normal", &mut button.normal);
            colour(ui, "Hover", &mut button.hover);
            colour(ui, "Pressed", &mut button.pressed);
            colour(ui, "Disabled", &mut button.disabled);
            widgets::property(ui, "Interactable", |ui| {
                ui.checkbox(&mut button.interactable, "").on_hover_text(
                "Off, it draws its disabled tint and nothing at or under it gets pointer events",
            );
            });
        },
    );
    edit(state, entity, BUTTON, &before, &button);
}
