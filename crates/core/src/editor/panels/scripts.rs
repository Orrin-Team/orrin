//! Scripting panel: swap in a freshly built game assembly without restarting,
//! and watch what rebuild-on-save is doing.
//!
//! The button only *requests* a reload — the swap destroys and re-creates
//! managed objects, so `App` runs it at the start of the next script phase
//! rather than from inside the UI pass. The watcher's auto-reload toggle takes
//! the same route: it decides whether a green build raises that request, never
//! whether a swap happens here.

use std::time::Duration;

use orrin_ecs::World;

use crate::build_watcher::{BuildState, BuildStatus};
use crate::editor::icons;
use crate::editor::state::EditorState;
use crate::editor::theme::{ERROR, OK, PENDING};
use crate::scene::Scripts;
use crate::scripting::{ScriptTimings, TimingRow};

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    let mut live = 0usize;
    let mut faulted = 0usize;
    world.query::<&Scripts>().for_each(|_, scripts| {
        live += scripts.len();
        faulted += scripts.iter().filter(|script| script.faulted).count();
    });

    if ui.button("Reload scripts").clicked() {
        state.request_script_reload();
    }
    if let Some(mut status) = world.get_resource_mut::<BuildStatus>() {
        ui.checkbox(&mut status.auto_reload, "Reload after a successful rebuild");
        ui.separator();
        show_build_state(ui, &status);
    }
    ui.separator();
    ui.label(format!("{live} behaviour(s) attached"));
    if faulted > 0 {
        icons::labelled(
            ui,
            icons::warning(),
            ERROR,
            &format!("{faulted} faulted — a reload clears them"),
        );
    }
    ui.separator();
    timings(ui, world);
}

fn timings(ui: &mut egui::Ui, world: &World) {
    let Some(mut timings) = world.get_resource_mut::<ScriptTimings>() else {
        return;
    };
    ui.checkbox(&mut timings.enabled, "Time each script");
    ui.label(
        egui::RichText::new(
            "Time in managed code per Behaviour type, every hook included, in ms. \
             The numbers include the cost of measuring: two clock reads around \
             every call, which on many small scripts is much of what they cost.",
        )
        .weak()
        .small(),
    );
    if !timings.enabled {
        return;
    }

    let mut rows: Vec<&TimingRow> = timings
        .rows()
        .filter(|row| row.average() > Duration::ZERO)
        .collect();
    if rows.is_empty() {
        ui.weak("Nothing has run yet.");
        return;
    }
    rows.sort_by_key(|row| std::cmp::Reverse(row.average()));
    let ms = |time: Duration| format!("{:.3}", time.as_secs_f64() * 1e3);
    egui::Grid::new("script timings")
        .num_columns(3)
        .striped(true)
        .spacing([10.0, 2.0])
        .show(ui, |ui| {
            for heading in ["", "last", "avg"] {
                ui.label(egui::RichText::new(heading).weak().small());
            }
            ui.end_row();
            for row in rows {
                let short = row.type_name.split(',').next().unwrap_or(&row.type_name);
                ui.label(short).on_hover_text(&row.type_name);
                ui.monospace(ms(row.last_frame));
                ui.monospace(ms(row.average()));
                ui.end_row();
            }
        });
}

fn show_build_state(ui: &mut egui::Ui, status: &BuildStatus) {
    match &status.state {
        BuildState::Idle => {
            ui.weak("Watching for script changes.");
        }
        BuildState::Building => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Rebuilding…");
            });
        }
        BuildState::Succeeded => {
            let took = status
                .last_duration
                .map_or_else(String::new, |d| format!(" ({:.1}s)", d.as_secs_f32()));
            // A green build is not the same as a live session running it: with
            // auto-reload off the compiled code is ahead of the loaded code
            // until someone presses the button.
            if status.pending_reload {
                icons::labelled(
                    ui,
                    icons::refresh(),
                    PENDING,
                    &format!("New build ready{took}"),
                );
                ui.weak("Click Reload scripts to run it.");
            } else {
                icons::labelled(ui, icons::check(), OK, &format!("Build up to date{took}"));
            }
        }
        BuildState::Failed => {
            icons::labelled(
                ui,
                icons::warning(),
                ERROR,
                &format!("Build failed — {} diagnostic(s)", status.diagnostics.len()),
            );
            ui.weak("Still running the previous build.");
            egui::CollapsingHeader::new("Compiler output")
                .default_open(true)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(120.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            for line in &status.diagnostics {
                                ui.colored_label(ERROR, line);
                            }
                        });
                });
        }
        BuildState::Unavailable(reason) => {
            icons::labelled(ui, icons::warning(), ERROR, "Could not run the compiler.");
            ui.weak(reason);
            ui.weak("Still watching; the next save tries again.");
        }
        BuildState::Off(reason) => {
            ui.weak("Rebuild-on-save is off.");
            ui.weak(reason);
        }
    }
}
