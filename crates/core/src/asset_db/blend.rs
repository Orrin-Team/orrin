//! A `.blend` as a mesh asset: Blender, run headless, exports the file as a
//! GLB and the glTF mesh import makes the mesh, as Godot and Unity import one.
//! Saving in Blender reimports through the watcher, into the slot the mesh
//! already has. An export of a one-cube file took 0.56 s on an M5 Pro with
//! Blender 5.2.3, nearly all of it Blender starting.
//!
//! **One file is one mesh, by the glTF rule.** The export goes through the same
//! mesh import a `.glb` does, so merging, node transforms, mirrored winding,
//! tangents, the sidecar's `scale`, the thumbnail and the summary are all that
//! code's; there is no second mesh path to drift from it. What is exported is
//! [`EXPORT`]: objects visible in the viewport, modifiers applied, +Y up, and no
//! materials, images, animations or skins, none of which a mesh asset reads.
//! What Blender's viewport shows is what the engine draws.
//!
//! **Blender opens the bytes that were hashed.** `import` hands an importer the
//! bytes the cache key hashed, never a second read of a file that may be
//! mid-save. Blender has to open a file, so those bytes are written to a private
//! temporary `.blend` and Blender opens that, never the source in place. A
//! `.blend` is therefore imported alone: data it links from another `.blend` by
//! relative path is not part of the asset, and is not tracked as an input
//! either. The temporary directory goes on every way out, a panic included.
//!
//! **Nothing in the file runs.** `--disable-autoexec` is not optional: a
//! `.blend` from a cloned repository must not run the scripts embedded in it
//! because the editor indexed it. `--factory-startup` keeps the user's
//! preferences and add-ons out of the export, and `--python-exit-code 1` is
//! what makes a Python exception a failed import — without it Blender exits 0.
//! The output path is read from `sys.argv` after `--` rather than spliced into
//! the Python, so no file name can quote its way into it. The expression also
//! refuses to run with no file loaded, so a Blender that carried on past a file
//! it could not read would not export the factory scene's cube in its place.
//! All of this was checked against Blender 5.2.3 LTS, whose glTF exporter is
//! enabled under `--factory-startup`.
//!
//! **Failure is named, never a hang.** No Blender, one that exits non-zero, one
//! that writes nothing: each fails the import naming the file, with the tail of
//! what Blender printed. Its output goes to a file rather than a pipe, so a
//! chatty Blender cannot block on a full one, and one that has not finished
//! within [`TIMEOUT`] is killed. The database's rules then apply unchanged: a
//! failed first import draws the pink placeholder, a failed save keeps drawing
//! the last version that imported.
//!
//! **Finding Blender:** `ORRIN_BLENDER`, then the path the editor hands the
//! database ([`set_blender`](super::AssetDatabase::set_blender)), then an
//! installed one — `blender` on `PATH`, else Blender.app on a Mac or the newest
//! `Blender Foundation\Blender *` under Program Files on Windows. The import
//! thread resolves it as each export starts, so a path set mid-session applies
//! to everything still queued.
//!
//! **The key** ([`key_settings`]) hashes [`EXPORT`] and the glTF importer's
//! version beside the sidecar's settings, under [`IMPORTER`], so changing the
//! export or either importer reimports every `.blend` and no `.gltf`. The
//! installed Blender's version is deliberately not in it: an upgrade does not
//! reimport the project — Reimport does — and a `.blend` already in the cache
//! loads with no Blender installed at all. An exported game never runs Blender:
//! it loads the artifact its pack carries, as it does any mesh's.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use super::cache::Importer;
use super::sidecar::Settings;

pub const IMPORTER: Importer = Importer {
    name: "blend-mesh",
    version: 1,
};

/// The Python Blender runs over the opened file, exporting it to the path after
/// `--`. What it exports is a contract with the live link, which streams the
/// same thing.
pub const EXPORT: &str = concat!(
    "import bpy, sys\n",
    "if not bpy.data.filepath: raise RuntimeError(\"the file did not open as a .blend\")\n",
    "bpy.ops.export_scene.gltf(filepath=sys.argv[sys.argv.index(\"--\") + 1], ",
    "export_format=\"GLB\", export_apply=True, use_visible=True, export_yup=True, ",
    "export_materials=\"NONE\", export_image_format=\"NONE\", export_animations=False, ",
    "export_skins=False, export_morph=False)",
);

/// How long an export may run before Blender is killed. An export is well under
/// a second, most of it Blender starting, so two minutes leaves room for a
/// heavy modifier stack while a Blender that is stuck holds the one import
/// thread for no longer than this.
pub const TIMEOUT: Duration = Duration::from_secs(120);

const FIX: &str = "Set Blender's path in Settings ▸ Blender, or in ORRIN_BLENDER";

/// How many of the last lines Blender printed a failure carries.
const TAIL: usize = 20;

/// How the import thread runs Blender.
#[derive(Clone, Debug, PartialEq)]
pub struct Blender {
    /// The path the editor was given, which `ORRIN_BLENDER` overrides.
    pub configured: Option<PathBuf>,
    pub timeout: Duration,
}

impl Default for Blender {
    fn default() -> Self {
        Self {
            configured: None,
            timeout: TIMEOUT,
        }
    }
}

pub fn is_blend(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("blend"))
}

/// The settings as a `.blend`'s cache key hashes them.
pub fn key_settings(settings: &Settings) -> String {
    format!(
        "{};gltf-mesh={};export={EXPORT}",
        settings.canonical(),
        super::import::MESH.version
    )
}

/// The Blender to run: `env` (`ORRIN_BLENDER`), else `configured`, else one
/// installed where Blender installs. A named path is returned whether or not it
/// exists, so the failure to start it names it.
pub fn locate(env: Option<OsString>, configured: Option<&Path>) -> Option<PathBuf> {
    env.filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            configured
                .filter(|path| !path.as_os_str().is_empty())
                .map(Path::to_owned)
        })
        .or_else(installed)
}

fn installed() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "blender.exe"
    } else {
        "blender"
    };
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|exe| exe.is_file())
        .or_else(platform_install)
}

#[cfg(target_os = "macos")]
fn platform_install() -> Option<PathBuf> {
    let exe = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    exe.is_file().then_some(exe)
}

#[cfg(windows)]
fn platform_install() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("ProgramFiles")?).join("Blender Foundation");
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let version = version_of(entry.file_name().to_str()?)?;
            let exe = entry.path().join("blender.exe");
            exe.is_file().then_some((version, exe))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, exe)| exe)
}

#[cfg(not(any(target_os = "macos", windows)))]
fn platform_install() -> Option<PathBuf> {
    None
}

/// `Blender 4.10` as `[4, 10]`, which sorts above `Blender 4.9` as a name would
/// not.
#[cfg(any(windows, test))]
fn version_of(dir: &str) -> Option<Vec<u32>> {
    dir.strip_prefix("Blender ")?
        .split('.')
        .map(|part| part.parse().ok())
        .collect()
}

/// `bytes`, the `.blend` at `rel`, exported by Blender as GLB bytes.
pub fn export(rel: &str, bytes: &[u8], blender: &Blender) -> Result<Vec<u8>, String> {
    let failed = |why: String| format!("{rel} could not be exported: {why}");
    let exe = locate(
        std::env::var_os("ORRIN_BLENDER"),
        blender.configured.as_deref(),
    )
    .ok_or_else(|| failed(format!("no Blender was found to export it. {FIX}")))?;
    let scratch = Scratch::new().map_err(failed)?;
    let copy = scratch.0.join("source.blend");
    let out = scratch.0.join("export.glb");
    let log = scratch.0.join("blender.log");
    let written = |path: &Path, error: std::io::Error| {
        failed(format!("{} could not be written: {error}", path.display()))
    };
    std::fs::write(&copy, bytes).map_err(|error| written(&copy, error))?;
    let output = std::fs::File::create(&log).map_err(|error| written(&log, error))?;
    let errors = output.try_clone().map_err(|error| written(&log, error))?;

    let mut child = Command::new(&exe)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
        ])
        .arg(&copy)
        .arg("--python-expr")
        .arg(EXPORT)
        .arg("--")
        .arg(&out)
        .current_dir(&scratch.0)
        .stdin(Stdio::null())
        .stdout(output)
        .stderr(errors)
        .spawn()
        .map_err(|error| {
            failed(format!(
                "Blender would not start from {}: {error}. {FIX}",
                exe.display()
            ))
        })?;

    let deadline = Instant::now() + blender.timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            waited => {
                let _ = child.kill();
                let _ = child.wait();
                let why = match waited {
                    Err(error) => format!("Blender could not be waited for: {error}"),
                    _ => format!(
                        "Blender did not finish within {} s and was stopped",
                        blender.timeout.as_secs_f32()
                    ),
                };
                return Err(failed(format!("{why}.{}", printed(&log))));
            }
        }
    };
    if !status.success() {
        return Err(failed(format!(
            "Blender exited with {status}.{}",
            printed(&log)
        )));
    }
    std::fs::read(&out).map_err(|_| {
        failed(format!(
            "Blender finished without writing a GLB.{}",
            printed(&log)
        ))
    })
}

/// The last lines Blender printed, for a message.
fn printed(log: &Path) -> String {
    let text = std::fs::read(log).unwrap_or_default();
    let text = String::from_utf8_lossy(&text);
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    match &lines[lines.len().saturating_sub(TAIL)..] {
        [] => " It printed nothing".to_owned(),
        tail => format!(" It printed:\n{}", tail.join("\n")),
    }
}

/// A directory of one export's own, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "orrin-blend-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // One left by an earlier process of the same id could hold a GLB that
        // would be read as this export's.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("{} could not be created: {error}", dir.display()))?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use orrin_registry::AssetId;

    use super::*;
    use crate::asset_db::import::tests::triangle_glb;
    use crate::asset_db::tests::{Dir, WAIT, drawing, handle, update};
    use crate::asset_db::{AssetDatabase, AssetKind, LoadState, Status, Target};
    use crate::gfx::{HeadlessBackend, RenderBackend};
    use crate::scene::CpuMesh;

    /// `dir` indexed with `blender` in place before anything is queued, so no
    /// import runs with whatever this machine has installed.
    fn opened(dir: &Dir, blender: Blender) -> AssetDatabase {
        let (mut db, issues) = AssetDatabase::empty(
            dir.path().to_owned(),
            dir.path().join(".cache"),
            Target::Rgba8,
        );
        assert!(issues.is_empty(), "{issues:?}");
        db.configure_blender(blender);
        let issues = db.scan(&[]);
        assert!(issues.is_empty(), "{issues:?}");
        db
    }

    fn configured(path: PathBuf) -> Blender {
        Blender {
            configured: Some(path),
            ..Blender::default()
        }
    }

    /// Wait for the import thread and take what it finished.
    fn settle(db: &mut AssetDatabase) {
        assert!(db.wait_idle(WAIT));
        db.update(&orrin_ecs::World::new(), &mut HeadlessBackend::new());
    }

    fn failure(db: &AssetDatabase, path: &str) -> String {
        match &db.find(path).unwrap().status {
            Status::Failed(message) => message.clone(),
            other => panic!("{path} is {other:?}"),
        }
    }

    /// A Blender stand-in in `tools` that runs `body`, after writing the
    /// arguments it was given to `tools/args`, NUL-separated.
    #[cfg(unix)]
    fn stand_in(tools: &Dir, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let args = tools.path().join("args");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\0' \"$@\" > '{}'\n{body}\n",
            args.display()
        );
        let path = tools.write("blender", script.as_bytes());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    fn args(tools: &Dir) -> Vec<String> {
        let args = std::fs::read_to_string(tools.path().join("args")).unwrap();
        args.split_terminator('\0').map(str::to_owned).collect()
    }

    /// One that writes [`triangle_glb`] wherever it is told to.
    #[cfg(unix)]
    fn exporting(tools: &Dir) -> PathBuf {
        let glb = tools.write("triangle.glb", &triangle_glb());
        stand_in(
            tools,
            &format!(
                "while [ \"$#\" -gt 1 ]; do\n  if [ \"$1\" = \"--\" ]; then cp '{}' \"$2\"; exit 0; fi\n  shift\ndone\nexit 3",
                glb.display()
            ),
        )
    }

    /// Blender is given a private copy of the bytes that were hashed, never the
    /// source, with autoexec off, and the copy is gone once it has finished.
    #[cfg(unix)]
    #[test]
    fn a_blend_imports_as_the_mesh_blender_exported_from_a_private_copy() {
        let tools = Dir::new("blend-tools-import");
        let dir = Dir::new("blend-import");
        let source = dir.write("models/tree.blend", b"BLENDER-v502 not read here");
        let db = opened(&dir, configured(exporting(&tools)));
        let entry = db.find("models/tree.blend").expect("a .blend is an asset");
        assert_eq!(entry.kind, AssetKind::Mesh);
        let id = entry.id;
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();

        update(&world, &mut backend);
        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        let errors = update(&world, &mut backend);
        assert!(errors.is_empty(), "{errors:?}");

        assert_eq!(
            world.resource::<AssetDatabase>().state(id),
            LoadState::Ready
        );
        let triangle =
            crate::geom::Aabb::from_points([glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y]);
        assert_eq!(backend.mesh_bounds(handle(&world, entity)), Some(triangle));

        let args = args(&tools);
        let [b, f, y, c, one, copy, e, expr, dash, out] = &args[..] else {
            panic!("{args:?}");
        };
        assert_eq!(
            [b, f, y, c, one, e, dash].map(String::as_str),
            [
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python-expr",
                "--"
            ]
        );
        assert_eq!(expr, EXPORT);
        assert!(out.ends_with(".glb"), "{out}");
        let copy = Path::new(copy);
        assert_ne!(copy, source);
        assert!(copy.extension().is_some_and(|e| e == "blend"), "{copy:?}");
        assert!(
            !copy.parent().unwrap().exists(),
            "the private copy is cleaned up"
        );
    }

    /// A failed first import draws the placeholder, and the message carries
    /// what Blender printed.
    #[cfg(unix)]
    #[test]
    fn a_blender_that_fails_is_named_with_what_it_printed() {
        let tools = Dir::new("blend-tools-fail");
        let blender = stand_in(
            &tools,
            "echo 'Read blend: somewhere'\necho 'RuntimeError: the exporter fell over' >&2\nexit 1",
        );
        let dir = Dir::new("blend-fail");
        dir.write("models/tree.blend", b"BLENDER");
        let db = opened(&dir, configured(blender));
        let id = db.find("models/tree.blend").unwrap().id;
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();
        update(&world, &mut backend);
        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        let errors = update(&world, &mut backend);

        let message = failure(&world.resource::<AssetDatabase>(), "models/tree.blend");
        assert!(
            message.starts_with("models/tree.blend could not be exported: Blender exited with"),
            "{message}"
        );
        assert!(
            message.contains("RuntimeError: the exporter fell over"),
            "{message}"
        );
        // Twice, when the scan's import and the one binding the slot queued
        // both fail; never anything else.
        assert!(
            !errors.is_empty() && errors.iter().all(|error| *error == message),
            "{errors:?}"
        );
        assert_eq!(
            backend.mesh_bounds(handle(&world, entity)),
            Some(CpuMesh::cube().bounds()),
            "the placeholder"
        );
        let copy = PathBuf::from(&args(&tools)[5]);
        assert!(
            !copy.parent().unwrap().exists(),
            "cleaned up on failure too"
        );
    }

    #[test]
    fn a_blender_that_is_not_there_names_where_to_set_it() {
        let dir = Dir::new("blend-missing");
        dir.write("models/tree.blend", b"BLENDER");
        let nowhere = dir.path().join("nowhere").join("blender");
        let mut db = opened(&dir, configured(nowhere));
        settle(&mut db);

        let message = failure(&db, "models/tree.blend");
        assert!(
            message.starts_with("models/tree.blend could not be exported"),
            "{message}"
        );
        assert!(
            message.contains("Settings ▸ Blender") && message.contains("ORRIN_BLENDER"),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_blender_that_never_finishes_is_stopped_and_the_import_fails() {
        let tools = Dir::new("blend-tools-hang");
        let pid = tools.path().join("pid");
        let blender = stand_in(
            &tools,
            &format!("echo $$ > '{}'\nexec sleep 30", pid.display()),
        );
        let dir = Dir::new("blend-hang");
        dir.write("models/tree.blend", b"BLENDER");
        let started = Instant::now();
        let mut db = opened(
            &dir,
            Blender {
                configured: Some(blender),
                timeout: Duration::from_millis(300),
            },
        );
        settle(&mut db);
        assert!(started.elapsed() < Duration::from_secs(10));

        let message = failure(&db, "models/tree.blend");
        assert!(
            message.starts_with(
                "models/tree.blend could not be exported: Blender did not finish within 0.3 s"
            ),
            "{message}"
        );
        let pid = std::fs::read_to_string(&pid).unwrap();
        let alive = std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
            .success();
        assert!(!alive, "Blender is stopped, not left running");
    }

    /// Fixing the path imports what failed for the want of it, with no restart.
    #[cfg(unix)]
    #[test]
    fn setting_blender_queues_every_blend_that_failed_again() {
        let tools = Dir::new("blend-tools-set");
        let dir = Dir::new("blend-set");
        dir.write("models/tree.blend", b"BLENDER");
        let mut db = opened(&dir, configured(dir.path().join("nowhere")));
        settle(&mut db);
        let id: AssetId = db.find("models/tree.blend").unwrap().id;
        assert_eq!(db.state(id), LoadState::Failed);

        db.set_blender(Some(exporting(&tools)));
        settle(&mut db);
        assert!(matches!(
            db.find("models/tree.blend").unwrap().status,
            Status::Imported { cached: false, .. }
        ));
        assert_eq!(db.state(id), LoadState::Ready);
    }

    #[test]
    fn the_newest_blender_install_is_the_highest_version_not_the_last_name() {
        let newest = [
            "Blender 4.9",
            "Blender 4.10",
            "Blender 3.6",
            "Blender Launcher",
        ]
        .into_iter()
        .max_by_key(|dir| version_of(dir));
        assert_eq!(newest, Some("Blender 4.10"));
        assert_eq!(version_of("Blender Launcher"), None);
    }

    /// The real thing: a 2 m cube moved 3 m up Blender's Z, beside a sphere
    /// hidden in the viewport, arrives as the cube alone, 3 m up Y.
    #[test]
    #[ignore = "needs Blender"]
    fn a_blend_saved_by_blender_imports_as_what_its_viewport_shows() {
        let blender = locate(std::env::var_os("ORRIN_BLENDER"), None)
            .expect("no Blender found; set ORRIN_BLENDER");
        let dir = Dir::new("blend-real");
        let source = dir.path().join("models").join("crate.blend");
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        let made = std::process::Command::new(&blender)
            .args([
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
            ])
            .arg("--python-expr")
            .arg(
                "import bpy, sys\n\
                 bpy.data.objects['Cube'].location = (0, 0, 3)\n\
                 bpy.ops.mesh.primitive_uv_sphere_add(location=(5, 0, 0))\n\
                 bpy.context.active_object.hide_set(True)\n\
                 bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index('--') + 1])",
            )
            .arg("--")
            .arg(&source)
            .output()
            .unwrap();
        assert!(made.status.success(), "{made:?}");

        let bytes = std::fs::read(&source).unwrap();
        let started = Instant::now();
        let glb = export("models/crate.blend", &bytes, &configured(blender.clone())).unwrap();
        eprintln!(
            "Blender exported {} bytes of .blend to {} bytes of GLB in {:?}",
            bytes.len(),
            glb.len(),
            started.elapsed()
        );

        let db = opened(&dir, configured(blender));
        let id = db.find("models/crate.blend").unwrap().id;
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();
        let started = Instant::now();
        update(&world, &mut backend);
        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        let errors = update(&world, &mut backend);
        eprintln!("imported through the database in {:?}", started.elapsed());
        assert!(errors.is_empty(), "{errors:?}");

        let summary = world
            .resource::<AssetDatabase>()
            .find("models/crate.blend")
            .unwrap()
            .summary;
        assert!(
            matches!(
                summary,
                Some(crate::asset_db::cache::Summary::Mesh { triangles: 12, .. })
            ),
            "{summary:?}"
        );
        let bounds = backend.mesh_bounds(handle(&world, entity)).unwrap();
        assert_eq!(bounds.min, glam::Vec3A::new(-1.0, 2.0, -1.0));
        assert_eq!(bounds.max, glam::Vec3A::new(1.0, 4.0, 1.0));
    }
}
