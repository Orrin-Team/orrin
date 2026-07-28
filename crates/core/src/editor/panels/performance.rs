use orrin_ecs::World;

use crate::stats::FrameStats;

pub fn show(ctx: &egui::Context, world: &World) {
    let Some(stats) = world.get_resource::<FrameStats>() else {
        return;
    };

    const CPU_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 200, 255);
    const GPU_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 170, 80);

    egui::Window::new("Performance")
        .default_pos(egui::pos2(12.0, 12.0))
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0} FPS", stats.fps()))
                    .size(24.0)
                    .strong(),
            );
            ui.colored_label(CPU_COLOR, format!("CPU: {:.2} ms (avg)", stats.frame_ms()));
            match stats.gpu_ms() {
                Some(ms) => ui.colored_label(GPU_COLOR, format!("GPU: {ms:.2} ms")),
                None => ui.colored_label(GPU_COLOR, "GPU: n/a"),
            };

            if !stats.history().is_empty() {
                let (min, max) = stats.min_max_ms();
                ui.label(format!("CPU min {min:.2} · max {max:.2} ms"));
            }

            let rss_mb = stats.memory_bytes() as f64 / (1024.0 * 1024.0);
            ui.label(format!("Memory (RSS): {rss_mb:.1} MB"));
            let total_gb = stats.vram_total() as f64 / (1024.0 * 1024.0 * 1024.0);
            match stats.vram_used() {
                Some(used) => {
                    let used_mb = used as f64 / (1024.0 * 1024.0);
                    ui.label(format!("VRAM: {used_mb:.0} MB / {total_gb:.1} GB"));
                }
                None => {
                    ui.label(format!("VRAM: {total_gb:.1} GB (usage n/a)"));
                }
            }

            ui.add_space(4.0);
            graph(ui, &stats, CPU_COLOR, GPU_COLOR);
        });
}

// Reference lines mark 60 fps (16.7 ms) and 30 fps (33.3 ms).
fn graph(ui: &mut egui::Ui, stats: &FrameStats, cpu_color: egui::Color32, gpu_color: egui::Color32) {
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

    for (ms, color) in [
        (16.67, egui::Color32::from_rgb(60, 120, 60)),
        (33.33, egui::Color32::from_rgb(130, 95, 40)),
    ] {
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
