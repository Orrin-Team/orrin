//! `orrin export`: a project, as a directory a player can run.
//!
//! ```text
//! space-shooter-mac/
//!   space-shooter        the engine, built with the `export` feature, stripped
//!   Orrin.dll            the bindings, and their runtimeconfig
//!   SpaceShooter.dll     the game assembly, Release
//!   dotnet/              a private .NET runtime: no SDK needed to play
//!   compiled-assets/     artifacts, sidecars and text assets; no sources
//!   orrin.json           what the engine reads to know which game it is
//! ```
//!
//! **Every step runs for the target, from the host.** The engine is
//! cross-compiled by `cargo xwin` (Windows) or `cargo zigbuild` (Linux) when the
//! target is not the machine exporting; a Mac game needs Apple's SDK, which is
//! only on a Mac. The C# is published for the target's runtime identifier, and
//! the runtime itself comes out of the NuGet runtime pack `dotnet publish
//! --self-contained` restores for that identifier — so a Windows runtime is
//! assembled on a Mac without a Windows machine in sight.
//!
//! **Assets are packed by the engine checkout's own build, on the host**,
//! because the game's executable may be for another platform and cannot run
//! here. The pack holds what the cache already has — see
//! `orrin_core::asset_db::pack` — so exporting twice from one import gives the
//! same bytes.
//!
//! **The runtime is a `DOTNET_ROOT`, not a self-contained publish.** See
//! `orrin_script::APP_LOCAL_RUNTIME` for why. A runtime pack has no
//! `Microsoft.NETCore.App.deps.json`, and hostfxr lists a framework only where
//! it finds one, so [`assemble_runtime`] writes it: every managed file under
//! `runtime`, every native one under `native`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use orrin_project::Project;
use orrin_project::export::{ExportManifest, Platform};

use crate::{Fallible, display_path, engine};

/// What a platform is built as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub platform: Platform,
    pub triple: &'static str,
    /// The .NET runtime identifier.
    pub rid: &'static str,
}

impl Target {
    /// Mac exports are Apple Silicon: that is where the engine's Metal backend
    /// is the default, and an Intel Mac would need MoltenVK shipped beside it.
    pub fn of(platform: Platform) -> Self {
        let (triple, rid) = match platform {
            Platform::Mac => ("aarch64-apple-darwin", "osx-arm64"),
            Platform::Linux => ("x86_64-unknown-linux-gnu", "linux-x64"),
            Platform::Windows => ("x86_64-pc-windows-msvc", "win-x64"),
        };
        Self {
            platform,
            triple,
            rid,
        }
    }
}

/// What compiles the engine for a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builder {
    Cargo,
    /// `cargo xwin`: the MSVC CRT and Windows SDK, downloaded, on any host.
    Xwin,
    /// `cargo zigbuild`: zig as the C and C++ cross compiler.
    Zigbuild,
}

impl Builder {
    /// The cargo subcommand, and how to install it when it is missing.
    fn subcommand(self) -> (&'static str, Option<&'static str>) {
        match self {
            Self::Cargo => ("build", None),
            Self::Xwin => ("xwin", Some("cargo install --locked cargo-xwin")),
            Self::Zigbuild => (
                "zigbuild",
                Some("cargo install --locked cargo-zigbuild, and install zig (brew install zig)"),
            ),
        }
    }

    /// The arguments to `cargo` that compile. `xwin` wraps cargo's own
    /// subcommands and takes `build` after it; `zigbuild` is the build.
    pub fn build_args(self) -> &'static [&'static str] {
        match self {
            Self::Cargo => &["build"],
            Self::Xwin => &["xwin", "build"],
            Self::Zigbuild => &["zigbuild"],
        }
    }
}

/// How `target` is built on `host`, or why it cannot be. `linux_sysroot` is
/// whether the environment names a Linux system root to link against.
pub fn builder(
    target: Platform,
    host: Option<Platform>,
    linux_sysroot: bool,
) -> Result<Builder, String> {
    if host == Some(target) {
        return Ok(Builder::Cargo);
    }
    match target {
        Platform::Windows => Ok(Builder::Xwin),
        Platform::Linux if linux_sysroot => Ok(Builder::Zigbuild),
        Platform::Linux => Err(
            "a Linux game links libudev, for gamepads, and this machine has no Linux system \
             libraries to link it against. Export it on Linux, take the `linux` package from the \
             CI export job, or set PKG_CONFIG_SYSROOT_DIR to a Linux sysroot that has libudev"
                .to_owned(),
        ),
        Platform::Mac => Err(
            "a Mac game can only be built on a Mac: it links against Apple's SDK, which is not \
             available anywhere else. Export it on a Mac, or take the `mac` package from the \
             CI export job"
                .to_owned(),
        ),
    }
}

/// Lay out a `DOTNET_ROOT` in `out` from a self-contained publish in `bundled`,
/// leaving out every file a framework-dependent publish of the same project in
/// `app` also has — what is left is the runtime pack.
pub fn assemble_runtime(bundled: &Path, app: &Path, rid: &str, out: &Path) -> Fallible {
    let config = bundled.join("Orrin.runtimeconfig.json");
    let version = framework_version(&config)?;

    let mut runtime = Vec::new();
    for entry in read_dir(bundled)? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_ours = app.join(&name).exists() || name.starts_with("Orrin.");
        if entry.path().is_file() && !is_ours {
            runtime.push(name);
        }
    }
    runtime.sort();

    let is_hostfxr = |name: &str| name.starts_with("hostfxr.") || name.starts_with("libhostfxr.");
    let Some(hostfxr) = runtime.iter().find(|name| is_hostfxr(name)) else {
        return Err(format!(
            "the .NET runtime pack for {rid} has no hostfxr in {}",
            display_path(bundled)
        ));
    };
    let fxr = out.join("host/fxr").join(&version);
    let framework = out.join("shared/Microsoft.NETCore.App").join(&version);
    create_dir(&fxr)?;
    create_dir(&framework)?;
    copy(&bundled.join(hostfxr), &fxr.join(hostfxr))?;

    let mut managed = BTreeMap::new();
    let mut native = BTreeMap::new();
    for name in runtime.iter().filter(|name| !is_hostfxr(name)) {
        copy(&bundled.join(name), &framework.join(name))?;
        let list = if is_assembly(&bundled.join(name)) {
            &mut managed
        } else {
            &mut native
        };
        list.insert(name.clone(), serde_json::json!({}));
    }

    let tfm = ".NETCoreApp,Version=v10.0";
    let library = format!("Microsoft.NETCore.App.Runtime.{rid}/{version}");
    let deps = serde_json::json!({
        "runtimeTarget": { "name": format!("{tfm}/{rid}"), "signature": "" },
        "compilationOptions": {},
        "targets": {
            tfm: {},
            format!("{tfm}/{rid}"): {
                &library: { "runtime": managed, "native": native }
            }
        },
        "libraries": {
            &library: { "type": "runtimepack", "serviceable": false, "sha512": "" }
        }
    });
    write(
        &framework.join("Microsoft.NETCore.App.deps.json"),
        &serde_json::to_string_pretty(&deps).unwrap_or_default(),
    )?;
    write(
        &framework.join("Microsoft.NETCore.App.runtimeconfig.json"),
        "{\n  \"runtimeOptions\": {\n    \"tfm\": \"net10.0\"\n  }\n}\n",
    )
}

/// Whether `path` is a .NET assembly: a PE image whose CLI header directory is
/// filled. Asked of the file rather than its name, because a Windows runtime
/// pack's native libraries are `.dll`s too and the set changes between
/// releases. Anything that is not a PE — a `.dylib`, a `.so` — is native.
fn is_assembly(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let half = |at: usize| {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    if bytes.get(..2) != Some(b"MZ") {
        return false;
    }
    let Some(pe) = word(0x3c).map(|at| at as usize) else {
        return false;
    };
    if bytes.get(pe..pe + 4) != Some(b"PE\0\0") {
        return false;
    }
    let optional = pe + 24;
    let directories = match half(optional) {
        Some(0x20b) => optional + 112,
        Some(0x10b) => optional + 96,
        _ => return false,
    };
    const CLI_HEADER: usize = 14;
    word(directories + CLI_HEADER * 8 + 4).is_some_and(|size| size > 0)
}

/// The runtime version a self-contained publish bundled, from its
/// runtimeconfig's `includedFrameworks`.
fn framework_version(config: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(config)
        .map_err(|error| format!("could not read {}: {error}", display_path(config)))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("could not parse {}: {error}", display_path(config)))?;
    json["runtimeOptions"]["includedFrameworks"]
        .as_array()
        .and_then(|frameworks| {
            frameworks
                .iter()
                .find(|framework| framework["name"] == "Microsoft.NETCore.App")
        })
        .and_then(|framework| framework["version"].as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "{} names no bundled Microsoft.NETCore.App; the publish was not self-contained",
                display_path(config)
            )
        })
}

/// Where `scene` (an absolute path in `project`) sits in the export: inside
/// the pack when it is an asset, at its own relative path otherwise. The second
/// half is the file to copy there, for a scene the pack does not carry.
pub fn scene_in_export(
    project: &Project,
    scene: &Path,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    if let Ok(rel) = scene.strip_prefix(project.assets_dir()) {
        return Ok((Path::new(PACK).join(rel), None));
    }
    match scene.strip_prefix(project.root()) {
        Ok(rel) => Ok((rel.to_path_buf(), Some(scene.to_path_buf()))),
        Err(_) => Err(format!(
            "{} is outside the project, so it cannot be exported",
            scene.display()
        )),
    }
}

/// Make `dir` an empty directory to export into. A previous export there is
/// replaced; anything else is refused, since it is not ours to delete.
pub fn prepare_out(dir: &Path) -> Fallible {
    if dir.exists() {
        let earlier = dir.join(orrin_project::export::FILE_NAME).is_file();
        let empty = read_dir(dir)?.is_empty();
        if earlier {
            std::fs::remove_dir_all(dir).map_err(|error| {
                format!(
                    "could not clear the earlier export in {}: {error}",
                    display_path(dir)
                )
            })?;
        } else if !empty {
            return Err(format!(
                "{} already holds files that are not an Orrin export; pick an empty directory \
                 with --out",
                display_path(dir)
            ));
        }
    }
    create_dir(dir)
}

/// The pack's directory inside an export.
const PACK: &str = "compiled-assets";

pub struct Options {
    pub platform: Option<Platform>,
    pub out: Option<PathBuf>,
}

pub fn export(project: &Project, options: &Options) -> Fallible {
    let host = Platform::host();
    let platform = options.platform.or(host).ok_or_else(|| {
        format!(
            "this machine ({}) is not a platform games export for; pass --platform",
            std::env::consts::OS
        )
    })?;
    let target = Target::of(platform);
    let builder = builder(
        platform,
        host,
        std::env::var_os("PKG_CONFIG_SYSROOT_DIR").is_some(),
    )?;
    let Some(checkout) = engine::find_checkout(project.root()) else {
        return Err(
            "exporting builds the engine from source, and no Orrin checkout was found above \
             this project or this CLI. Run the `orrin` built inside a checkout"
                .to_owned(),
        );
    };
    let entry = project.entry_type();
    let Some(assembly) = entry
        .split_once(',')
        .map(|(_, assembly)| assembly.trim())
        .filter(|assembly| !assembly.is_empty())
    else {
        return Err(format!(
            "`scripts.entry` is `{entry}`, which names no assembly; write it as `Namespace.Type, \
             Assembly`"
        ));
    };
    let out = options.out.clone().unwrap_or_else(|| {
        project
            .root()
            .join("exports")
            .join(format!("{}-{platform}", project.name()))
    });

    // Everything that can fail in a second, before anything that takes minutes.
    preflight(&checkout, builder, target)?;
    let staging = project.editor_dir().join("export").join(platform.name());
    let _ = std::fs::remove_dir_all(&staging);
    let mut manifest = ExportManifest::new(project, platform, format!("{assembly}.dll"));
    let mut loose_scenes = Vec::new();
    for (scene, slot) in [
        (project.start_scene(), &mut manifest.start_scene),
        (project.persistent_scene(), &mut manifest.persistent_scene),
    ] {
        if let Some(scene) = scene {
            let (inside, copy) = scene_in_export(project, &scene)?;
            if let Some(copy) = copy {
                loose_scenes.push((copy, inside.clone()));
            }
            *slot = Some(inside);
        }
    }

    step(1, "publishing the scripts", &target);
    let bindings_project = checkout.join("scripting/Orrin");
    let bindings = staging.join("bindings");
    let runtime = staging.join("runtime");
    let game = staging.join("game");
    publish(&bindings_project, target.rid, false, &bindings, None)?;
    publish(&bindings_project, target.rid, true, &runtime, None)?;
    publish(
        &project.scripts_dir(),
        target.rid,
        false,
        &game,
        Some(&bindings),
    )?;

    step(2, "building the engine", &target);
    let engine_exe = build_engine(&checkout, builder, target)?;

    step(3, "packing the assets", &target);
    prepare_out(&out)?;
    pack_assets(&checkout, project, &out.join(PACK))?;

    step(4, "assembling the game", &target);
    let exe = out.join(format!("{}{}", project.name(), platform.exe_suffix()));
    copy(&engine_exe, &exe)?;
    copy_published(&bindings, &out, |_| true)?;
    copy_published(&game, &out, |name| !name.starts_with("Orrin."))?;
    assemble_runtime(&runtime, &bindings, target.rid, &out.join(RUNTIME))?;
    for (from, inside) in loose_scenes {
        copy(&from, &out.join(inside))?;
    }
    let input = project.root().join("input.toml");
    if input.is_file() {
        copy(&input, &out.join("input.toml"))?;
    }
    manifest
        .write(&out)
        .map_err(|error| format!("could not write orrin.json: {error}"))?;

    println!(
        "orrin: exported `{}` for {platform} -> {} ({})",
        project.name(),
        display_path(&out),
        size_label(directory_size(&out))
    );
    if host == Some(platform) {
        println!("orrin: play it with {}", display_path(&exe));
    } else {
        println!("orrin: copy the whole directory to a {platform} machine to play it");
    }
    Ok(())
}

/// `orrin_script::APP_LOCAL_RUNTIME`, spelled again because the CLI does not
/// link the script host.
const RUNTIME: &str = "dotnet";

fn step(n: u32, what: &str, target: &Target) {
    println!(
        "orrin: [{n}/4] {what} for {} ({})",
        target.platform, target.triple
    );
}

/// Check every tool the export will need, so a missing one is named at once
/// rather than after a ten-minute compile.
fn preflight(checkout: &Path, builder: Builder, target: Target) -> Fallible {
    let found = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .output()
            .is_ok_and(|out| out.status.success())
    };
    if !found("dotnet", &["--version"]) {
        return Err("`dotnet` is not installed or not on PATH; install the .NET 10 SDK".to_owned());
    }
    let (subcommand, install) = builder.subcommand();
    // `--help`, not `--version`: `cargo zigbuild` refuses the latter.
    if let Some(install) = install
        && !found("cargo", &[subcommand, "--help"])
    {
        return Err(format!(
            "building for {} from this machine needs `cargo {subcommand}`. Install it with: \
             {install}",
            target.platform
        ));
    }
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .current_dir(checkout)
        .output();
    match installed {
        Ok(out)
            if String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|t| t == target.triple) => {}
        Ok(_) => {
            println!("orrin: adding the Rust target {}", target.triple);
            run(Command::new("rustup")
                .args(["target", "add", target.triple])
                .current_dir(checkout))?;
        }
        // No rustup: a toolchain installed some other way. Cargo will say if
        // it lacks the target.
        Err(_) => {}
    }
    Ok(())
}

fn publish(
    project: &Path,
    rid: &str,
    self_contained: bool,
    out: &Path,
    bindings: Option<&Path>,
) -> Fallible {
    let mut command = Command::new("dotnet");
    command
        .arg("publish")
        .arg(project)
        .args(["-c", "Release", "-r", rid, "--nologo", "-v", "quiet"])
        .arg(if self_contained {
            "--self-contained"
        } else {
            "--no-self-contained"
        })
        .arg("-o")
        .arg(out);
    if let Some(bindings) = bindings {
        command.arg(format!("-p:OrrinBindings={}", bindings.display()));
    }
    run(&mut command).map_err(|error| format!("publishing {}: {error}", display_path(project)))
}

/// Build the engine for `target` in release, with the `export` feature and no
/// symbols. Returns the executable.
/// The arguments after `cargo` that build the engine for `target`.
///
/// `--release` is the workspace's most optimised profile — level 3, ThinLTO,
/// one codegen unit — and the symbols are stripped on top. Fat LTO was
/// measured against it with `tests/perf.rs` at `ORRIN_STRESS=4000` and made no
/// difference to any CPU phase, so a slower link buys nothing here.
fn engine_args(builder: Builder, target: Target) -> Vec<String> {
    builder
        .build_args()
        .iter()
        .copied()
        .chain(["--release", "-p", "orrin-core", "--features", "export"])
        .chain(["--target", target.triple])
        .map(str::to_owned)
        .collect()
}

/// Where cargo leaves the engine `engine_args` built.
fn engine_exe(checkout: &Path, target: Target) -> PathBuf {
    checkout
        .join("target")
        .join(target.triple)
        .join("release")
        .join(format!("orrin-core{}", target.platform.exe_suffix()))
}

fn build_engine(checkout: &Path, builder: Builder, target: Target) -> Result<PathBuf, String> {
    let mut command = Command::new("cargo");
    // From the checkout, so its `rust-toolchain.toml` picks the compiler and
    // the target `preflight` installed is the one this build looks for.
    command
        .current_dir(checkout)
        .args(engine_args(builder, target))
        .arg("--manifest-path")
        .arg(checkout.join("Cargo.toml"))
        .env("CARGO_PROFILE_RELEASE_STRIP", "symbols")
        .env("CARGO_PROFILE_RELEASE_DEBUG", "false");
    // shaderc's vendored CMake files predate CMake 4; see the CI workflow.
    if std::env::var_os("CMAKE_POLICY_VERSION_MINIMUM").is_none() {
        command.env("CMAKE_POLICY_VERSION_MINIMUM", "3.5");
    }
    run(&mut command).map_err(|error| format!("building the engine: {error}"))?;
    let exe = engine_exe(checkout, target);
    if !exe.is_file() {
        return Err(format!(
            "the engine built, but {} is not where cargo should have put it",
            display_path(&exe)
        ));
    }
    Ok(exe)
}

/// The checkout's own engine, on this machine, importing what the project's
/// cache lacks and writing the pack.
fn pack_assets(checkout: &Path, project: &Project, out: &Path) -> Fallible {
    let mut command = Command::new("cargo");
    command
        .current_dir(checkout)
        .args(["run", "--quiet", "-p", "orrin-core", "--manifest-path"])
        .arg(checkout.join("Cargo.toml"))
        .arg("--")
        .arg("--pack-assets")
        .arg(project.root())
        .arg(out);
    run(&mut command).map_err(|error| format!("packing the assets: {error}"))
}

/// Copy a publish's output into the export, without debug symbols and without
/// the files `keep` refuses.
fn copy_published(from: &Path, to: &Path, keep: impl Fn(&str) -> bool) -> Fallible {
    for entry in read_dir(from)? {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_file() && !name.ends_with(".pdb") && keep(&name) {
            copy(&entry.path(), &to.join(&name))?;
        }
    }
    Ok(())
}

fn run(command: &mut Command) -> Fallible {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command.status().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            format!("`{program}` is not installed or not on PATH")
        } else {
            format!("could not run `{program}`: {error}")
        }
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "`{program}` failed ({status}); its output is above"
        ))
    }
}

fn read_dir(dir: &Path) -> Result<Vec<std::fs::DirEntry>, String> {
    std::fs::read_dir(dir)
        .and_then(Iterator::collect)
        .map_err(|error| format!("could not read {}: {error}", display_path(dir)))
}

fn create_dir(dir: &Path) -> Fallible {
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("could not create {}: {error}", display_path(dir)))
}

fn copy(from: &Path, to: &Path) -> Fallible {
    if let Some(dir) = to.parent() {
        create_dir(dir)?;
    }
    std::fs::copy(from, to)
        .map(|_| ())
        .map_err(|error| format!("could not copy {}: {error}", display_path(from)))
}

fn write(path: &Path, text: &str) -> Fallible {
    std::fs::write(path, text)
        .map_err(|error| format!("could not write {}: {error}", display_path(path)))
}

fn directory_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => directory_size(&entry.path()),
            _ => entry.metadata().map_or(0, |meta| meta.len()),
        })
        .sum()
}

fn size_label(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    /// The smallest PE32+ image the classification reads: DOS stub pointer,
    /// signature, optional header magic, and a CLI data directory that is
    /// filled for a managed assembly and empty for a native library.
    fn pe(path: &Path, managed: bool) {
        let at = 0x80;
        let mut bytes = vec![0u8; at + 24 + 112 + 16 * 8];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[0x3c..0x40].copy_from_slice(&(at as u32).to_le_bytes());
        bytes[at..at + 4].copy_from_slice(b"PE\0\0");
        bytes[at + 24..at + 26].copy_from_slice(&0x20bu16.to_le_bytes());
        if managed {
            let cli = at + 24 + 112 + 14 * 8;
            bytes[cli..cli + 4].copy_from_slice(&0x2000u32.to_le_bytes());
            bytes[cli + 4..cli + 8].copy_from_slice(&72u32.to_le_bytes());
        }
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn project() -> (tempfile::TempDir, Project) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("orrin.toml"),
            "format_version = 1\n[project]\nname = \"g\"\nversion = \"1\"\n\
             [scripts]\ndir = \"scripts\"\nentry = \"G.Main, G\"\n",
        )
        .unwrap();
        let project = Project::load(&dir.path().join("orrin.toml")).unwrap();
        (dir, project)
    }

    #[test]
    fn each_platform_names_its_triple_and_runtime() {
        assert_eq!(
            Target::of(Platform::Windows),
            Target {
                platform: Platform::Windows,
                triple: "x86_64-pc-windows-msvc",
                rid: "win-x64"
            }
        );
        assert_eq!(Target::of(Platform::Linux).rid, "linux-x64");
        assert_eq!(Target::of(Platform::Mac).triple, "aarch64-apple-darwin");
    }

    #[test]
    fn the_host_builds_itself_and_cross_targets_name_their_tool() {
        for platform in Platform::ALL {
            assert_eq!(builder(platform, Some(platform), false), Ok(Builder::Cargo));
        }
        assert_eq!(
            builder(Platform::Windows, Some(Platform::Mac), false),
            Ok(Builder::Xwin)
        );
        assert_eq!(
            builder(Platform::Linux, Some(Platform::Mac), true),
            Ok(Builder::Zigbuild)
        );
        assert_eq!(
            builder(Platform::Linux, Some(Platform::Windows), true),
            Ok(Builder::Zigbuild)
        );
    }

    /// The engine links libudev for gamepads, and zig carries no Linux system
    /// libraries, so without a sysroot the build would fail a minute in on a
    /// pkg-config error. It is refused before anything starts instead.
    #[test]
    fn a_linux_game_from_elsewhere_without_a_sysroot_is_refused_with_the_way_out() {
        let error = builder(Platform::Linux, Some(Platform::Mac), false).unwrap_err();
        assert!(error.contains("libudev"), "{error}");
        assert!(error.contains("PKG_CONFIG_SYSROOT_DIR"), "{error}");
        assert!(error.contains("CI"), "{error}");
    }

    #[test]
    fn each_builder_is_invoked_the_way_its_tool_expects() {
        assert_eq!(Builder::Cargo.build_args(), ["build"]);
        assert_eq!(Builder::Xwin.build_args(), ["xwin", "build"]);
        assert_eq!(Builder::Zigbuild.build_args(), ["zigbuild"]);
    }

    /// Exports ship the release build with the `export` feature, and the
    /// executable is looked for where that build writes it.
    #[test]
    fn the_engine_is_built_for_release_with_the_export_feature() {
        let windows = Target::of(Platform::Windows);
        let args = engine_args(Builder::Xwin, windows);
        assert_eq!(&args[..2], ["xwin", "build"]);
        let after = |flag: &str| {
            let at = args
                .iter()
                .position(|arg| arg == flag)
                .unwrap_or_else(|| panic!("no {flag}: {args:?}"));
            args[at + 1].clone()
        };
        assert!(args.iter().any(|arg| arg == "--release"), "{args:?}");
        assert_eq!(after("--features"), "export");
        assert_eq!(after("--target"), "x86_64-pc-windows-msvc");

        assert_eq!(
            engine_exe(Path::new("/orrin"), windows),
            Path::new("/orrin/target/x86_64-pc-windows-msvc/release/orrin-core.exe")
        );
        assert_eq!(
            engine_exe(Path::new("/orrin"), Target::of(Platform::Mac)),
            Path::new("/orrin/target/aarch64-apple-darwin/release/orrin-core")
        );
    }

    #[test]
    fn a_mac_game_from_another_machine_is_refused_with_the_reason() {
        let error = builder(Platform::Mac, Some(Platform::Linux), false).unwrap_err();
        assert!(error.contains("Mac"), "{error}");
        assert!(error.contains("CI"), "says where it can be built: {error}");
    }

    #[test]
    fn the_runtime_is_the_bundled_publish_less_the_app_laid_out_as_a_dotnet_root() {
        let dir = tempfile::tempdir().unwrap();
        let (bundled, app, out) = (
            dir.path().join("sc"),
            dir.path().join("fd"),
            dir.path().join("dotnet"),
        );
        for file in [
            "Orrin.dll",
            "Orrin.deps.json",
            "libcoreclr.dylib",
            "libhostfxr.dylib",
            "createdump",
        ] {
            touch(&bundled.join(file));
        }
        pe(&bundled.join("System.Private.CoreLib.dll"), true);
        std::fs::write(
            bundled.join("Orrin.runtimeconfig.json"),
            r#"{"runtimeOptions":{"tfm":"net10.0","includedFrameworks":[{"name":"Microsoft.NETCore.App","version":"10.0.9"}]}}"#,
        )
        .unwrap();
        for file in ["Orrin.dll", "Orrin.deps.json", "Orrin.runtimeconfig.json"] {
            touch(&app.join(file));
        }

        assemble_runtime(&bundled, &app, "osx-arm64", &out).unwrap();

        assert!(out.join("host/fxr/10.0.9/libhostfxr.dylib").is_file());
        let framework = out.join("shared/Microsoft.NETCore.App/10.0.9");
        for runtime in [
            "System.Private.CoreLib.dll",
            "libcoreclr.dylib",
            "createdump",
        ] {
            assert!(framework.join(runtime).is_file(), "{runtime}");
        }
        for app_file in ["Orrin.dll", "Orrin.deps.json", "libhostfxr.dylib"] {
            assert!(
                !framework.join(app_file).exists(),
                "{app_file} is not the framework's"
            );
        }
        let deps: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(framework.join("Microsoft.NETCore.App.deps.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            deps["runtimeTarget"]["name"],
            ".NETCoreApp,Version=v10.0/osx-arm64"
        );
        let pack = &deps["targets"][".NETCoreApp,Version=v10.0/osx-arm64"]["Microsoft.NETCore.App.Runtime.osx-arm64/10.0.9"];
        assert!(
            pack["runtime"]["System.Private.CoreLib.dll"].is_object(),
            "{deps}"
        );
        assert!(pack["native"]["libcoreclr.dylib"].is_object(), "{deps}");
        assert!(
            framework
                .join("Microsoft.NETCore.App.runtimeconfig.json")
                .is_file()
        );
    }

    /// A Windows runtime's native libraries are `.dll`s too, so the file name
    /// cannot say which list a file belongs in; its PE header can.
    #[test]
    fn a_windows_runtime_is_split_by_what_each_dll_is_not_by_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let (bundled, out) = (dir.path().join("sc"), dir.path().join("dotnet"));
        pe(&bundled.join("System.Private.CoreLib.dll"), true);
        pe(&bundled.join("System.Net.Quic.dll"), true);
        for native in ["coreclr.dll", "msquic.dll", "mscorrc.dll", "hostfxr.dll"] {
            pe(&bundled.join(native), false);
        }
        touch(&bundled.join("createdump.exe"));
        std::fs::write(
            bundled.join("Orrin.runtimeconfig.json"),
            r#"{"runtimeOptions":{"includedFrameworks":[{"name":"Microsoft.NETCore.App","version":"10.0.9"}]}}"#,
        )
        .unwrap();

        assemble_runtime(&bundled, &dir.path().join("fd"), "win-x64", &out).unwrap();

        assert!(out.join("host/fxr/10.0.9/hostfxr.dll").is_file());
        let framework = out.join("shared/Microsoft.NETCore.App/10.0.9");
        let deps: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(framework.join("Microsoft.NETCore.App.deps.json")).unwrap(),
        )
        .unwrap();
        let pack = &deps["targets"][".NETCoreApp,Version=v10.0/win-x64"]["Microsoft.NETCore.App.Runtime.win-x64/10.0.9"];
        let names = |list: &str| -> Vec<String> {
            let mut names: Vec<String> = pack[list].as_object().unwrap().keys().cloned().collect();
            names.sort();
            names
        };
        assert_eq!(
            names("runtime"),
            ["System.Net.Quic.dll", "System.Private.CoreLib.dll"]
        );
        assert_eq!(
            names("native"),
            ["coreclr.dll", "createdump.exe", "mscorrc.dll", "msquic.dll"]
        );
    }

    #[test]
    fn a_publish_that_is_not_self_contained_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let bundled = dir.path().join("sc");
        std::fs::create_dir_all(&bundled).unwrap();
        std::fs::write(
            bundled.join("Orrin.runtimeconfig.json"),
            r#"{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}"#,
        )
        .unwrap();

        let error = assemble_runtime(&bundled, dir.path(), "osx-arm64", &dir.path().join("out"))
            .unwrap_err();
        assert!(error.contains("Orrin.runtimeconfig.json"), "{error}");
    }

    #[test]
    fn a_scene_in_the_assets_is_found_in_the_pack_and_any_other_is_copied() {
        let (dir, project) = project();
        let root = project.root().to_path_buf();

        let asset = root.join("assets/levels/one.orrin");
        assert_eq!(
            scene_in_export(&project, &asset),
            Ok((PathBuf::from("compiled-assets/levels/one.orrin"), None))
        );

        let loose = root.join("scenes/main.orrin");
        assert_eq!(
            scene_in_export(&project, &loose),
            Ok((PathBuf::from("scenes/main.orrin"), Some(loose.clone())))
        );
        drop(dir);
    }

    #[test]
    fn only_an_empty_directory_or_an_earlier_export_is_exported_into() {
        let dir = tempfile::tempdir().unwrap();

        let fresh = dir.path().join("fresh");
        prepare_out(&fresh).unwrap();
        assert!(fresh.is_dir());

        let earlier = dir.path().join("earlier");
        touch(&earlier.join(orrin_project::export::FILE_NAME));
        touch(&earlier.join("stale.dll"));
        prepare_out(&earlier).unwrap();
        assert!(
            !earlier.join("stale.dll").exists(),
            "an earlier export is replaced"
        );

        let theirs = dir.path().join("theirs");
        touch(&theirs.join("thesis.docx"));
        let error = prepare_out(&theirs).unwrap_err();
        assert!(error.contains("--out"), "{error}");
        assert!(theirs.join("thesis.docx").exists(), "never deleted");
    }
}
