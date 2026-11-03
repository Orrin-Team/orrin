// A shipped game is a windowed program: on Windows the console subsystem would
// open a terminal beside it. Its log goes to a file instead (`logging`).
#![cfg_attr(all(windows, feature = "export"), windows_subsystem = "windows")]

use orrin_core::App;

fn main() {
    // `orrin export`'s half of packing a game, run on the engine checkout's own
    // build. Not in an exported game, which has nothing to import from.
    #[cfg(not(feature = "export"))]
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let [flag, project, out] = &args[..]
            && flag == "--pack-assets"
        {
            std::process::exit(orrin_core::asset_db::pack::run_tool(
                std::path::Path::new(project),
                std::path::Path::new(out),
            ));
        }
    }
    App::run();
}
