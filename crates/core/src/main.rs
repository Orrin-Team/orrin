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
        // `orrin test`'s half: the project's tests, with no window. Not in an
        // exported game, which ships no tests to run.
        if args.first().is_some_and(|flag| flag == "--test") {
            #[cfg(feature = "scripting")]
            std::process::exit(orrin_core::testing::run_tool(&args[1..]));
            #[cfg(not(feature = "scripting"))]
            {
                eprintln!("orrin-core: built without scripting, so there are no tests to run");
                std::process::exit(2);
            }
        }
    }
    App::run();
}
