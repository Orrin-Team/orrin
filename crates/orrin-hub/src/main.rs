//! Orrin Hub: create, open and export projects without a terminal.
//!
//! A window over the `orrin` CLI and nothing more — every button runs an
//! `orrin` command, so the hub can never do something the CLI does not, and
//! a user who outgrows it already knows the commands it was running.

mod app;
mod recents;
mod task;
mod theme;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Orrin Hub")
            .with_inner_size([1180.0, 720.0])
            .with_min_inner_size([860.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Orrin Hub",
        options,
        Box::new(|cc| Ok(Box::new(app::Hub::new(cc)))),
    )
}
