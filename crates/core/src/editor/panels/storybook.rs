//! Storybook panel: the game's `[Story]` methods, grouped by the first part of
//! their names, beside the open story's controls.
//!
//! Choosing a story only writes `Storybook::open`; the app builds it, with the
//! script host, before the next frame draws, and shows it in a window of its
//! own until it is closed — here, or by closing that window. A moved control
//! sets `rebuild` the same way, so a story is always what its code builds from
//! its current controls, never a tree patched to match them.
//!
//! The panel lives in the bottom dock by default, which is wide and short, so
//! the list and the open story sit side by side and each scrolls on its own: a
//! story with a dozen controls never pushes the list out of the tab. Only a
//! dock too narrow for two usable columns stacks them, list first.

use orrin_ecs::World;

use crate::editor::icons;
use crate::editor::theme::{ERROR, KIND_UI};
use crate::editor::widgets::{self, Tool};
use crate::ui::story::{StoryStatus, StoryValue, Storybook};

/// A list row, the hierarchy's height, so the two lists read alike.
const ROW: f32 = 24.0;
/// The width at which the list and the open story go side by side.
const SPLIT_AT: f32 = 420.0;
const LIST_MIN: f32 = 170.0;
const LIST_MAX: f32 = 280.0;
/// The least the open story's column is left, which a slider and its value
/// still fit in.
const DETAIL_MIN: f32 = 220.0;
/// The widest a control's label column grows before its names are cut.
const LABEL_MAX: f32 = 140.0;

pub fn body(ui: &mut egui::Ui, world: &mut World) {
    let Some(mut book) = world.get_resource_mut::<Storybook>() else {
        return;
    };
    if !book.available {
        widgets::padded(ui, |ui| {
            ui.add_space(6.0);
            ui.weak("Stories are built by the game's scripts, and scripting is not running.");
        });
        return;
    }
    let area = ui.available_rect_before_wrap();
    match list_width(area.width()) {
        Some(width) => {
            let (list, detail) = area.split_left_right_at_x(area.left() + width);
            column(ui, list, |ui| story_list(ui, &mut book));
            ui.painter().vline(
                detail.left().round() + 0.5,
                detail.y_range(),
                ui.visuals().widgets.noninteractive.bg_stroke,
            );
            column(ui, detail.shrink2(egui::vec2(1.0, 0.0)), |ui| {
                open_story(ui, &mut book)
            });
        }
        None => {
            let (list, detail) = area.split_top_bottom_at_fraction(0.5);
            column(ui, list, |ui| story_list(ui, &mut book));
            column(ui, detail, |ui| open_story(ui, &mut book));
        }
    }
    ui.allocate_rect(area, egui::Sense::hover());
}

/// The width of the list when the panel is wide enough to split, or `None`.
fn list_width(width: f32) -> Option<f32> {
    (width >= SPLIT_AT).then(|| {
        (width * 0.38)
            .clamp(LIST_MIN, LIST_MAX)
            .min(width - DETAIL_MIN)
    })
}

/// `add` inside `rect`, in a child of its own so each half scrolls alone.
fn column(ui: &mut egui::Ui, rect: egui::Rect, add: impl FnOnce(&mut egui::Ui)) {
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    child.set_clip_rect(rect.intersect(ui.clip_rect()));
    add(&mut child);
}

fn story_list(ui: &mut egui::Ui, book: &mut Storybook) {
    let id = ui.id().with("storybook_filter");
    let mut query = ui
        .data(|data| data.get_temp::<String>(id))
        .unwrap_or_default();
    ui.add_space(4.0);
    widgets::padded(ui, |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| widgets::search(ui, &mut query, "Filter stories"),
        );
    });
    ui.add_space(4.0);

    if book.stories.is_empty() {
        widgets::padded(ui, |ui| {
            ui.label("No stories yet");
            ui.weak("Mark a static method in the game's scripts with");
            ui.label(egui::RichText::new("[Story(\"Menus/Pause\")]").monospace());
            ui.weak("and it is listed here after the next build.");
        });
        ui.data_mut(|data| data.insert_temp(id, query));
        return;
    }
    let groups = groups(&book.stories, &query);
    let searching = !query.trim().is_empty();
    if groups.is_empty() {
        widgets::padded(ui, |ui| {
            ui.weak(format!("No story matches “{}”.", query.trim()));
        });
    }

    let mut chosen = None;
    ui.spacing_mut().item_spacing.y = 0.0;
    egui::ScrollArea::vertical()
        .id_salt("storybook_list")
        .auto_shrink(false)
        .show(ui, |ui| {
            for (group, stories) in &groups {
                let open_id = ui.id().with(("storybook_group", *group));
                let open = searching || ui.data(|data| data.get_temp(open_id)).unwrap_or(true);
                if group_row(ui, group, stories.len(), open, searching).clicked() {
                    ui.data_mut(|data| data.insert_temp(open_id, !open));
                }
                if !open {
                    continue;
                }
                for (leaf, name) in stories {
                    let selected = book.open.as_deref() == Some(*name);
                    if story_row(ui, leaf, selected).clicked() {
                        chosen = Some((*name).to_owned());
                    }
                }
            }
        });
    if let Some(name) = chosen {
        if book.open.as_deref() == Some(&name) {
            book.rebuild = true;
        } else {
            book.open = Some(name);
        }
    }
    ui.data_mut(|data| data.insert_temp(id, query));
}

/// A group's disclosure row: chevron, name, and — while it is closed, so its
/// stories are out of sight — how many it holds.
fn group_row(
    ui: &mut egui::Ui,
    name: &str,
    count: usize,
    open: bool,
    locked: bool,
) -> egui::Response {
    let sense = if locked {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), sense);
    let visuals = ui.visuals();
    if response.hovered() && !locked {
        ui.painter()
            .rect_filled(rect, 0.0, visuals.widgets.hovered.weak_bg_fill);
    }
    let weak = visuals.weak_text_color();
    let chevron = egui::Rect::from_center_size(
        egui::pos2(rect.left() + widgets::PAD + 6.0, rect.center().y),
        egui::Vec2::splat(12.0),
    );
    let glyph = if open {
        icons::chevron_down()
    } else {
        icons::chevron_right()
    };
    icons::paint(ui, glyph, chevron, weak);
    let text = egui::pos2(chevron.right() + 6.0, rect.center().y);
    let painted = ui.painter().text(
        text,
        egui::Align2::LEFT_CENTER,
        name,
        egui::TextStyle::Body.resolve(ui.style()),
        visuals.strong_text_color(),
    );
    if !open {
        ui.painter().text(
            egui::pos2(painted.right() + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            count.to_string(),
            egui::TextStyle::Small.resolve(ui.style()),
            weak,
        );
    }
    response
}

/// A story's row, indented under its group: the whole row is the target.
fn story_row(ui: &mut egui::Ui, name: &str, selected: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::click());
    let visuals = ui.visuals();
    let fill = if selected {
        Some(visuals.selection.bg_fill)
    } else if response.hovered() {
        Some(visuals.widgets.hovered.weak_bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter().rect_filled(rect, 0.0, fill);
    }
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + widgets::PAD + 26.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    icons::paint(ui, icons::frame_corners(), glyph, KIND_UI);
    let colour = if selected {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let text_rect = egui::Rect::from_min_max(
        egui::pos2(glyph.right() + 6.0, rect.top()),
        egui::pos2(rect.right() - widgets::PAD, rect.bottom()),
    );
    let galley = egui::WidgetText::from(name).into_galley(
        ui,
        Some(egui::TextWrapMode::Truncate),
        text_rect.width(),
        egui::TextStyle::Body,
    );
    ui.painter().galley(
        egui::pos2(text_rect.left(), rect.center().y - galley.size().y / 2.0),
        galley,
        colour,
    );
    response.on_hover_text_at_pointer(name)
}

fn open_story(ui: &mut egui::Ui, book: &mut Storybook) {
    let Some(name) = book.open.clone() else {
        widgets::padded(ui, |ui| {
            ui.add_space(10.0);
            ui.weak("Choose a story to open it in a window of its own.");
        });
        return;
    };
    ui.add_space(4.0);
    widgets::padded(ui, |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y + 2.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                if widgets::tool_button(
                    ui,
                    icons::x(),
                    None,
                    None,
                    Tool::Command,
                    "Close the story and its window",
                )
                .clicked()
                {
                    book.open = None;
                    book.args = Default::default();
                }
                if widgets::tool_button(
                    ui,
                    icons::refresh(),
                    None,
                    None,
                    Tool::Command,
                    "Build it again from its code, restarting its tweens",
                )
                .clicked()
                {
                    book.rebuild = true;
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    title(ui, &name);
                });
            },
        );
        match book.status {
            StoryStatus::Faulted => icons::labelled(
                ui,
                icons::warning(),
                ERROR,
                "It threw while building; the console has why",
            ),
            StoryStatus::Missing => icons::labelled(
                ui,
                icons::warning(),
                ERROR,
                "The game no longer declares this story",
            ),
            StoryStatus::Closed | StoryStatus::Built => {}
        }
    });
    ui.add_space(6.0);

    if book.args.controls().is_empty() {
        widgets::padded(ui, |ui| {
            ui.weak("No controls. A story asks for one with ctx.Float or ctx.Bool.");
        });
        return;
    }
    let label_width = book
        .args
        .controls()
        .iter()
        .map(|control| {
            egui::WidgetText::from(control.name.as_str())
                .into_galley(ui, None, f32::INFINITY, egui::TextStyle::Body)
                .size()
                .x
        })
        .fold(0.0, f32::max)
        .min(LABEL_MAX);

    let mut changed = false;
    egui::ScrollArea::vertical()
        .id_salt("storybook_controls")
        .auto_shrink(false)
        .show(ui, |ui| {
            widgets::padded(ui, |ui| {
                for control in book.args.controls_mut() {
                    ui.horizontal(|ui| {
                        ui.set_min_height(crate::editor::theme::CONTROL_HEIGHT);
                        let (label, _) = ui.allocate_exact_size(
                            egui::vec2(label_width, crate::editor::theme::CONTROL_HEIGHT),
                            egui::Sense::hover(),
                        );
                        let galley = egui::WidgetText::from(control.name.as_str()).into_galley(
                            ui,
                            Some(egui::TextWrapMode::Truncate),
                            label_width,
                            egui::TextStyle::Body,
                        );
                        ui.painter().galley(
                            egui::pos2(label.left(), label.center().y - galley.size().y / 2.0),
                            galley,
                            ui.visuals().text_color(),
                        );
                        ui.add_space(8.0);
                        match &mut control.value {
                            StoryValue::Float { value, min, max } => {
                                // The value box beside the slider is egui's own;
                                // the slider takes what is left of the column.
                                let value_box =
                                    ui.spacing().interact_size.x + ui.spacing().item_spacing.x;
                                ui.spacing_mut().slider_width =
                                    (ui.available_width() - value_box).max(60.0);
                                changed |= ui
                                    .add(
                                        egui::Slider::new(value, *min..=*max)
                                            .clamping(egui::SliderClamping::Always),
                                    )
                                    .changed();
                            }
                            StoryValue::Bool(value) => {
                                changed |= ui.checkbox(value, "").changed();
                            }
                        }
                    });
                    ui.add_space(4.0);
                }
            });
        });
    book.rebuild |= changed;
}

/// A story's name as its group, weak, then the rest of it.
fn title(ui: &mut egui::Ui, name: &str) {
    ui.spacing_mut().item_spacing.x = 4.0;
    match name.split_once('/') {
        Some((group, leaf)) => {
            ui.weak(group);
            ui.weak("/");
            ui.strong(leaf);
        }
        None => {
            ui.strong(name);
        }
    }
}

/// Stories whose names contain `query`, ignoring case, under the first part of
/// their names, in the order they came; a name with no slash is its own
/// group's only story. Each story is its name past the group, and its whole
/// name.
fn groups<'a>(stories: &'a [String], query: &str) -> Vec<(&'a str, Vec<(&'a str, &'a str)>)> {
    let query = query.trim().to_lowercase();
    let mut groups: Vec<(&str, Vec<(&str, &str)>)> = Vec::new();
    for name in stories {
        if !query.is_empty() && !name.to_lowercase().contains(&query) {
            continue;
        }
        let (group, leaf) = name.split_once('/').unwrap_or((name, name));
        match groups.iter_mut().find(|(existing, _)| *existing == group) {
            Some((_, stories)) => stories.push((leaf, name)),
            None => groups.push((group, vec![(leaf, name)])),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stories() -> Vec<String> {
        [
            "Buttons/Primary",
            "Buttons/Disabled",
            "Toast",
            "Menus/Pause/Settings",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn stories_are_grouped_by_the_first_part_of_their_names() {
        let stories = stories();
        assert_eq!(
            groups(&stories, ""),
            vec![
                (
                    "Buttons",
                    vec![
                        ("Primary", "Buttons/Primary"),
                        ("Disabled", "Buttons/Disabled")
                    ]
                ),
                ("Toast", vec![("Toast", "Toast")]),
                ("Menus", vec![("Pause/Settings", "Menus/Pause/Settings")]),
            ]
        );
    }

    /// A filter matches anywhere in the whole name, ignoring case, so typing a
    /// group's name lists the group.
    #[test]
    fn a_filter_keeps_the_stories_whose_names_contain_it() {
        let stories = stories();
        assert_eq!(
            groups(&stories, "  pAuSe "),
            vec![("Menus", vec![("Pause/Settings", "Menus/Pause/Settings")])]
        );
        assert_eq!(groups(&stories, "button").len(), 1);
        assert_eq!(groups(&stories, "button")[0].1.len(), 2);
        assert!(groups(&stories, "nothing like it").is_empty());
    }

    /// Side by side wherever the list and the controls both fit; the list takes
    /// a share of the width within bounds, so neither column collapses.
    #[test]
    fn the_panel_splits_when_it_is_wide_enough() {
        assert_eq!(list_width(300.0), None);
        let narrowest = list_width(SPLIT_AT).expect("wide enough to split");
        assert!(narrowest >= LIST_MIN && SPLIT_AT - narrowest >= DETAIL_MIN);
        assert_eq!(list_width(4000.0), Some(LIST_MAX));
    }
}
