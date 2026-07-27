//! Standalone lifecycle smoke test: boot CoreCLR, create a Behaviour, tick it.
//! Run with the cwd set to the built `Ferron.dll` directory
//! (`scripting/Ferron/bin/Debug/net10.0`).
//!
//! With no arguments it drives a probe that lives in the bindings assembly, so
//! it needs nothing but `dotnet build scripting/Ferron`. Pass a game DLL and a
//! type name to exercise the collectible-load-context path instead:
//!
//! ```text
//! cargo run -p ferron-script --example spike -- \
//!     ../../../DemoGame/bin/Debug/net10.0/DemoGame.dll "DemoGame.Spinner, DemoGame"
//! ```

use std::ffi::CString;
use std::path::Path;

use ferron_script::{default_api, CEntity, ScriptHost};

fn main() {
    let host = match ScriptHost::boot(&default_api(), Path::new(".")) {
        Ok(host) => host,
        Err(err) => {
            eprintln!("scripting host failed: {err}");
            std::process::exit(1);
        }
    };

    let mut args = std::env::args().skip(1);
    let type_name = match (args.next(), args.next()) {
        (Some(dll), Some(type_name)) => {
            // Same two-step swap the engine performs: stage the assembly in a
            // collectible context, then commit it (nothing to retire yet).
            let path = CString::new(dll.clone()).unwrap();
            let staged = host.stage_game(&path);
            if !staged.succeeded() {
                eprintln!("[host] could not stage {dll}: {staged}");
                std::process::exit(1);
            }
            println!("[host] committed game assembly: {}", host.commit_game());
            type_name
        }
        _ => "Ferron.Tests.LifecycleProbe, Ferron".to_owned(),
    };

    let type_name = CString::new(type_name).unwrap();
    let handle = host.create(CEntity::NULL, &type_name);
    println!("[host] created behaviour handle = {handle:#x}");

    host.enable(handle);
    host.start(handle);
    for _ in 0..3 {
        host.update(handle, 0.016);
    }
}
