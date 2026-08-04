use orrin_ecs::World;

use super::figures;
use crate::editor::theme;
use crate::profile::{self, Lane, Profiler, Row};
use crate::scene::Culling;
use crate::stats::FrameStats;

pub fn body(ui: &mut egui::Ui, world: &World) {
    let Some(stats) = world.get_resource::<FrameStats>() else {
        return;
    };
    ui.label(
        egui::RichText::new(format!("{:.0} FPS", stats.fps()))
            .size(24.0)
            .strong(),
    );
    ui.label(figures(format!("CPU: {:.2} ms (avg)", stats.frame_ms())).color(theme::CPU));
    ui.label(match stats.gpu_ms() {
        Some(ms) => figures(format!("GPU: {ms:.2} ms")).color(theme::GPU),
        None => figures("GPU: n/a").color(theme::GPU),
    });

    if !stats.history().is_empty() {
        let (min, max) = stats.min_max_ms();
        ui.label(figures(format!("CPU min {min:.2} · max {max:.2} ms")));
    }

    let rss_mb = stats.memory_bytes() as f64 / (1024.0 * 1024.0);
    ui.label(figures(format!("Memory (RSS): {rss_mb:.1} MB")));
    let total_gb = stats.vram_total() as f64 / (1024.0 * 1024.0 * 1024.0);
    match stats.vram_used() {
        Some(used) => {
            let used_mb = used as f64 / (1024.0 * 1024.0);
            ui.label(figures(format!("VRAM: {used_mb:.0} MB / {total_gb:.1} GB")));
        }
        None => {
            ui.label(figures(format!("VRAM: {total_gb:.1} GB (usage n/a)")));
        }
    }

    if let Some(mut culling) = world.get_resource_mut::<Culling>() {
        ui.label(figures(format!(
            "Draws: {} / {} ({} culled)",
            culling.visible(),
            culling.total(),
            culling.culled(),
        )));
        ui.checkbox(&mut culling.enabled, "Frustum culling")
            .on_hover_text("Off draws everything — the A/B for a suspected culling bug.");
    }

    ui.add_space(4.0);
    graph(ui, &stats, theme::CPU, theme::GPU);

    if let Some(profiler) = world.get_resource::<Profiler>() {
        ui.add_space(6.0);
        ui.separator();

        let mut enabled = profile::is_enabled();
        if ui.checkbox(&mut enabled, "Collect phase timings").changed() {
            profile::set_enabled(enabled);
        }
        if enabled {
            phases(ui, &profiler, Lane::Cpu, "CPU phases", theme::CPU);
            phases(ui, &profiler, Lane::Gpu, "GPU passes", theme::GPU);
        }
    }
}

/// One lane's phase table, slowest first.
fn phases(ui: &mut egui::Ui, profiler: &Profiler, lane: Lane, title: &str, color: egui::Color32) {
    let rows = profiler.aggregate(lane);
    egui::CollapsingHeader::new(egui::RichText::new(title).color(color))
        .default_open(true)
        .show(ui, |ui| {
            if rows.is_empty() {
                // The GPU lane is empty for the first frames of a run: readback
                // trails, so there is nothing to show rather than nothing to time.
                ui.weak("no spans yet");
                return;
            }

            let total = lane_total(&rows);
            egui::Grid::new(title)
                .num_columns(5)
                .striped(true)
                .spacing([10.0, 2.0])
                .show(ui, |ui| {
                    for heading in ["", "last", "avg", "max", "share"] {
                        ui.label(egui::RichText::new(heading).weak().small());
                    }
                    ui.end_row();

                    for row in &rows {
                        let indent = "  ".repeat(row.depth as usize);
                        ui.label(format!("{indent}{}", row.name));
                        ui.monospace(format!("{:.2}", row.last_ms));
                        ui.monospace(format!("{:.2}", row.avg_ms));
                        ui.monospace(format!("{:.2}", row.max_ms));
                        if total > 0.0 && row.depth == 0 {
                            ui.monospace(format!("{:.0}%", 100.0 * row.last_ms / total));
                        } else {
                            ui.label("");
                        }
                        ui.end_row();
                    }
                });
        });
}

/// What a share is a share *of*.
///
/// The GPU lane reserves a `frame` row spanning everything, so summing top-level
/// rows there would double-count it. Where no such row exists (the CPU lane), the
/// top-level rows are the total.
fn lane_total(rows: &[Row]) -> f32 {
    if let Some(frame) = rows.iter().find(|row| row.name == "frame") {
        return frame.last_ms;
    }
    rows.iter()
        .filter(|row| row.depth == 0)
        .map(|row| row.last_ms)
        .sum()
}

// Reference lines mark 60 fps (16.7 ms) and 30 fps (33.3 ms).
fn graph(
    ui: &mut egui::Ui,
    stats: &FrameStats,
    cpu_color: egui::Color32,
    gpu_color: egui::Color32,
) {
    let cpu = stats.history();
    let gpu = stats.gpu_history();
    let size = egui::vec2(ui.available_width().max(220.0), 64.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(128));

    // Scale so the worst recent CPU/GPU frame (or the 30 fps line, whichever is
    // larger) sits near the top — readable common case without clipping spikes.
    let peak = cpu
        .iter()
        .chain(gpu.iter())
        .copied()
        .fold(0.0_f32, f32::max);
    let scale_ms = peak.max(33.3);
    let y_for = |ms: f32| rect.bottom() - rect.height() * (ms / scale_ms).clamp(0.0, 1.0);

    for (ms, color) in [(16.67, theme::GUIDE_60), (33.33, theme::GUIDE_30)] {
        if ms <= scale_ms {
            painter.hline(rect.x_range(), y_for(ms), egui::Stroke::new(1.0, color));
        }
    }

    for (data, color) in [(cpu, cpu_color), (gpu, gpu_color)] {
        if data.len() >= 2 {
            let n = data.len();
            let points: Vec<egui::Pos2> = data
                .iter()
                .enumerate()
                .map(|(i, &ms)| {
                    let x = rect.left() + rect.width() * (i as f32 / (n - 1) as f32);
                    egui::pos2(x, y_for(ms))
                })
                .collect();
            painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
        }
    }
}
