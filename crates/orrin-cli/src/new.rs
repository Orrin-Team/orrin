use std::path::{Component, Path, PathBuf};

use crate::{Fallible, display_path, engine};

pub fn scaffold(name: &str, parent: &Path) -> Fallible {
    let assembly = assembly_name(name)?;

    if !parent.is_dir() {
        return Err(format!("{} is not a directory", parent.display()));
    }

    let root = parent.join(name);
    if root.exists() {
        let empty = std::fs::read_dir(&root).is_ok_and(|mut dir| dir.next().is_none());
        if !empty {
            return Err(format!(
                "{} already exists and is not empty",
                root.display()
            ));
        }
    }

    // The bindings reference differs inside and outside the engine's own tree,
    // so the checkout has to be resolved before any file is written. It is the
    // checkout containing the *new project*, never the one the CLI was built
    // in: the reference it produces gets committed.
    let anchor = parent
        .canonicalize()
        .map_err(|err| format!("could not resolve {}: {err}", parent.display()))?;
    let checkout = engine::checkout_containing(&anchor);

    let files = [
        ("orrin.toml", manifest(name, &assembly)),
        ("input.toml", INPUT.to_string()),
        (".gitignore", GITIGNORE.to_string()),
        ("README.md", readme(name, &assembly)),
        ("assets/.gitkeep", String::new()),
        ("scenes/.gitkeep", String::new()),
        (".vscode/launch.json", LAUNCH.to_string()),
        ("scripts/Main.cs", main_cs(name, &assembly)),
        ("scripts/FlyCamera.cs", fly_camera_cs(&assembly)),
        (
            &format!("scripts/{assembly}.csproj"),
            csproj(&assembly, checkout.as_deref(), &anchor.join(name)),
        ),
    ];

    for (relative, contents) in files {
        write(&root.join(relative), &contents)?;
    }

    println!("orrin: created `{name}` at {}", display_path(&root));
    println!();
    println!("  cd {}", display_path(&root));
    println!("  orrin run");
    Ok(())
}

fn write(path: &Path, contents: &str) -> Fallible {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("could not create {}: {err}", parent.display()))?;
    }
    std::fs::write(path, contents)
        .map_err(|err| format!("could not write {}: {err}", path.display()))
}

/// A project name doubles as a directory name and as the source of the game
/// assembly's name, so it has to be usable as both.
fn assembly_name(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("the project name is empty".to_string());
    }
    if Path::new(name).components().count() != 1
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(format!(
            "`{name}` is not a plain directory name; use `--path` to choose where the project goes"
        ));
    }

    let assembly: String = name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect();

    if assembly.is_empty() {
        return Err(format!(
            "`{name}` has no letters or digits to build an assembly name from"
        ));
    }
    if assembly.starts_with(|c: char| c.is_ascii_digit()) {
        return Err(format!(
            "`{name}` starts with a digit, which cannot begin a C# namespace"
        ));
    }

    Ok(assembly)
}

fn manifest(name: &str, assembly: &str) -> String {
    format!(
        r#"format_version = 1

[project]
name = "{name}"
version = "0.1.0"

[scripts]
dir = "scripts"
# Assembly-qualified: the half after the comma is both the namespace root and
# the DLL the engine loads, so it must match <AssemblyName> in the .csproj.
entry = "{assembly}.Main, {assembly}"

[assets]
dir = "assets"

# Uncomment once the scene format lands (issue #6):
# [scenes]
# start = "scenes/main.fscene"
"#
    )
}

/// The starting bindings. Present rather than left to the user because a
/// project without the file has no actions at all, and `Input.IsPressed("Jump")`
/// answering false forever is a poor first experience of a system whose whole
/// point is that controls are data.
const INPUT: &str = r#"format_version = 1

# What each control is called. A name may list several sources; any of them
# fires it. Keyboard keys are unprefixed; other devices are not.
[actions]
Jump    = ["Space", "Gamepad.South"]
Left    = ["A", "LeftArrow", "Gamepad.DPadLeft"]
Right   = ["D", "RightArrow", "Gamepad.DPadRight"]
Forward = ["W", "UpArrow", "Gamepad.DPadUp"]
Back    = ["S", "DownArrow", "Gamepad.DPadDown"]

# Axes compose the actions above, so a key is named once and rebinding an
# action moves every axis built on it. An axis may instead read one analog
# source directly, with a deadzone and a scale.
[axes.Horizontal]
positive = "Right"
negative = "Left"

[axes.Vertical]
positive = "Forward"
negative = "Back"

[axes.LookX]
source = "Mouse.X"
scale = 0.1

[axes.LookY]
source = "Mouse.Y"
scale = 0.1
"#;

const GITIGNORE: &str = "# Build output
bin/
obj/

# Engine-generated cache and asset index
.orrin/

# Games made by `orrin export`
exports/
";

/// The game assembly is loaded into the engine's own process, so there is no
/// game executable to launch under a debugger; both configurations attach.
/// `orrin-core` is the engine's name both from a checkout and installed.
const LAUNCH: &str = r#"{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to Orrin",
      "type": "coreclr",
      "request": "attach",
      "processName": "orrin-core"
    },
    {
      "name": "Pick a process to attach to",
      "type": "coreclr",
      "request": "attach",
      "processId": "${command:pickProcess}"
    }
  ]
}
"#;

fn readme(name: &str, assembly: &str) -> String {
    let csproj = format!("{assembly}.csproj");
    let tree = [
        (
            "  orrin.toml",
            "project manifest — the engine reads this from cwd (or any parent)",
        ),
        (
            "  input.toml",
            "action and axis bindings, reloaded when you save it",
        ),
        ("  assets/", "source assets + .meta sidecars"),
        ("  scripts/", "C# game code"),
        (
            &format!("    {csproj}"),
            &format!("builds {assembly}.dll, this project's game assembly"),
        ),
        (
            "    Main.cs",
            "the entry Behaviour named by `scripts.entry`",
        ),
        ("    FlyCamera.cs", "WASD and right-drag to move the camera"),
        ("  scenes/", "scene files"),
        (
            "  .vscode/launch.json",
            "attaches VS Code's C# debugger to the engine",
        ),
        ("  .orrin/", "generated, gitignored: cache + asset index"),
    ];

    // The assembly name sets the column, so the tree is aligned rather than
    // padded to a width that only happens to fit one project's name.
    let column = tree.iter().map(|(entry, _)| entry.len()).max().unwrap_or(0) + 2;
    let tree = tree
        .iter()
        .map(|(entry, description)| format!("{entry:column$}{description}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"# {name}

An Orrin project.

```
{name}/
{tree}
```

## Running

```
orrin run
```

That builds the scripts, indexes the assets, and launches the engine on this
project. `orrin build` stops after the build.

## Hot reload

With the engine running, edit `scripts/Main.cs`, then:

```
orrin build
```

and press **Reload scripts** in the editor's Scripts panel. Fields keep their
current values across the swap, so the scene picks up where it left off. A
build that fails leaves the running session untouched.

## Debugging

Your scripts run inside the engine's process, so a debugger attaches to the
running engine rather than launching the game. Start it with `orrin run` —
a Debug build, which keeps locals and line numbers; `--release` optimises them
away — then:

- **VS Code**, with the C# extension: Run and Debug, **Attach to Orrin**. It
  finds the engine by its process name, `orrin-core`; **Pick a process to
  attach to** asks instead.
- **Rider**: Run ▸ Attach to Process…, and pick `orrin-core`.

The engine loads `{assembly}.pdb` alongside `{assembly}.dll`, which is what lets
a breakpoint in `scripts/` bind to the code that is running.
"#
    )
}

/// Inside the engine's checkout the bindings are a sibling project; outside it
/// they are a shipped `Orrin.dll` whose location only the CLI knows, so it is
/// passed in as `OrrinBindings` (with `$ORRIN_SCRIPT_DIR` as the fallback for
/// a bare `dotnet build`). Either way `Private="false"` keeps `Orrin.dll` out
/// of this project's output: a second copy is a second set of types, and every
/// Behaviour cast across the boundary would fail.
///
/// The analyzer behind ORRIN001 follows the same split: a project reference in
/// the checkout, and beside `Orrin.dll` elsewhere, where it is optional so that
/// bindings shipped without it still build.
fn csproj(assembly: &str, checkout: Option<&Path>, root: &Path) -> String {
    let scripts_dir = root.join("scripts");
    let in_checkout = |project: &str| {
        let checkout = checkout?;
        let path = relative_path(&scripts_dir, &checkout.join(project))?;
        Some(path.to_string_lossy().replace('\\', "/"))
    };
    let reference = match (
        in_checkout("scripting/Orrin/Orrin.csproj"),
        in_checkout("scripting/Orrin.Analyzers/Orrin.Analyzers.csproj"),
    ) {
        (Some(bindings), Some(analyzers)) => format!(
            r#"    <ProjectReference Include="{bindings}" Private="false" />
    <!-- ORRIN001, a Behaviour field the registry cannot save. Loaded by the
         compiler, never referenced by the game assembly. -->
    <ProjectReference Include="{analyzers}"
                      OutputItemType="Analyzer" ReferenceOutputAssembly="false" />"#
        ),
        _ => r#"    <Reference Include="Orrin" Private="false">
      <HintPath>$(OrrinBindings)/Orrin.dll</HintPath>
    </Reference>
    <!-- ORRIN001, a Behaviour field the registry cannot save. -->
    <Analyzer Include="$(OrrinBindings)/Orrin.Analyzers.dll" Condition="Exists('$(OrrinBindings)/Orrin.Analyzers.dll')" />"#
            .to_string(),
    };

    format!(
        r#"<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <!-- The directory is `scripts`, which would otherwise name the assembly.
         This has to agree with the assembly half of `scripts.entry` in
         orrin.toml — that is where the engine reads the game DLL's name. -->
    <AssemblyName>{assembly}</AssemblyName>
    <RootNamespace>{assembly}</RootNamespace>
    <!-- The engine's host is already running the runtime; a game assembly is
         loaded into a collectible context and never boots its own. -->
    <GenerateRuntimeConfigurationFiles>false</GenerateRuntimeConfigurationFiles>
    <DebugType>portable</DebugType>
  </PropertyGroup>

  <PropertyGroup Condition="'$(Configuration)' == 'Debug'">
    <!-- Gates `Debug.DrawLine` and friends, which are [Conditional] on this;
         without it they compile away to nothing even in a Debug build. -->
    <DefineConstants>$(DefineConstants);ORRIN_DEBUG</DefineConstants>
  </PropertyGroup>

  <PropertyGroup Condition="'$(OrrinBindings)' == ''">
    <!-- `orrin build` passes this in; the env var is the fallback for a bare
         `dotnet build`. -->
    <OrrinBindings>$(ORRIN_SCRIPT_DIR)</OrrinBindings>
  </PropertyGroup>

  <ItemGroup>
{reference}
  </ItemGroup>

</Project>
"#
    )
}

fn main_cs(name: &str, assembly: &str) -> String {
    format!(
        r#"using Orrin;
using Orrin.Math;

namespace {assembly};

// The project's entry Behaviour, named by `scripts.entry` in orrin.toml. The
// engine attaches exactly this one at startup; everything else in the scene is
// spawned from here through the script API.
//
// Public fields are the project's knobs: they survive a hot reload, so you can
// change SpinSpeed, `orrin build`, hit "Reload scripts", and watch it take
// effect without restarting. Keep setup in OnStart — constructors run an extra
// time on reload to work out which fields you have actually changed.
public class Main : Behaviour
{{
    public float Height = 1.5f;

    public override void OnStart()
    {{
        var spawn = Transform.Identity;
        spawn.Position = new Vector3(0f, Height, 0f);
        var cube = World.SpawnRenderable("cube", "gold", spawn);
        World.AddScript<Spinner>(cube);
        World.AddScript<FlyCamera>(World.Spawn());
        Debug.Log($"{name} started; cube = {{cube}}");
    }}
}}

// Turns the entity it is attached to. A Behaviour moves its own entity, through
// LocalTransform, so the cube carries one of these rather than Main reaching
// over to turn it.
public class Spinner : Behaviour
{{
    public float SpinSpeed = 45f;

    private float _spin;

    public override void OnUpdate(float deltaTime)
    {{
        _spin += SpinSpeed * deltaTime;

        var t = LocalTransform;
        t.Rotation = Quaternion.Euler(0f, _spin, 0f);
        LocalTransform = t;
    }}
}}
"#
    )
}

fn fly_camera_cs(assembly: &str) -> String {
    format!(
        r#"using Orrin;
using Orrin.Math;

namespace {assembly};

// Walk the scene: WASD to move, hold the right mouse button and move the mouse
// to look, E and Q to rise and sink, Shift to go faster. Movement reads the
// axes input.toml defines, so rebinding a key there rebinds it here.
public class FlyCamera : Behaviour
{{
    public float Speed = 4f;
    public float Boost = 3f;
    public float LookSpeed = 1f;

    private float _yaw;
    private float _pitch;

    public override void OnStart()
    {{
        Camera.Place(new Vector3(0f, 1.5f, 5f), Quaternion.identity);
    }}

    public override void OnUpdate(float deltaTime)
    {{
        if (Input.GetMouseButton(MouseButton.Right))
        {{
            _yaw -= Input.GetAxis("LookX") * LookSpeed;
            _pitch = System.Math.Clamp(_pitch - Input.GetAxis("LookY") * LookSpeed, -89f, 89f);
        }}
        var facing = Quaternion.Euler(_pitch, _yaw, 0f);

        var move = facing * Vector3.forward * Input.GetAxis("Vertical")
            + facing * Vector3.right * Input.GetAxis("Horizontal");
        if (Input.GetKey(KeyCode.E)) move += Vector3.up;
        if (Input.GetKey(KeyCode.Q)) move -= Vector3.up;

        var speed = Speed * (Input.GetKey(KeyCode.LeftShift) ? Boost : 1f);
        Camera.Place(Camera.position + move * (speed * deltaTime), facing);
    }}
}}
"#
    )
}

/// `to`, expressed relative to `from_dir`. Both must be absolute; `None` when
/// they share no prefix, which on Unix only happens for malformed input.
fn relative_path(from_dir: &Path, to: &Path) -> Option<PathBuf> {
    let from: Vec<_> = from_dir.components().collect();
    let to: Vec<_> = to.components().collect();

    let shared = from
        .iter()
        .zip(to.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if shared == 0 {
        return None;
    }

    let mut path = PathBuf::new();
    path.extend(std::iter::repeat_n(
        Component::ParentDir,
        from.len() - shared,
    ));
    path.extend(&to[shared..]);
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_names_are_pascal_case() {
        assert_eq!(assembly_name("hello-orrin").unwrap(), "HelloOrrin");
        assert_eq!(assembly_name("my_game").unwrap(), "MyGame");
        assert_eq!(assembly_name("pong").unwrap(), "Pong");
        assert_eq!(assembly_name("Space Rocks").unwrap(), "SpaceRocks");
    }

    #[test]
    fn unusable_names_are_rejected() {
        for name in ["", "..", "a/b", "---", "2d-shooter"] {
            assert!(assembly_name(name).is_err(), "`{name}` should be rejected");
        }
    }

    #[test]
    fn relative_path_walks_up_and_back_down() {
        let from = Path::new("/repo/examples/game/scripts");
        let to = Path::new("/repo/scripting/Orrin/Orrin.csproj");
        assert_eq!(
            relative_path(from, to).unwrap(),
            Path::new("../../../scripting/Orrin/Orrin.csproj")
        );
    }

    #[test]
    fn a_scaffolded_project_loads() {
        let dir = tempfile::tempdir().expect("tempdir");
        scaffold("test-game", dir.path()).expect("scaffold");

        let root = dir.path().join("test-game");
        let project = orrin_project::Project::locate(&root)
            .expect("the generated manifest is valid")
            .expect("the generated manifest is found");

        assert_eq!(project.name(), "test-game");
        assert_eq!(project.entry_type(), "TestGame.Main, TestGame");
        assert_eq!(project.scripts_dir(), project.root().join("scripts"));
        assert_eq!(project.assets_dir(), project.root().join("assets"));
        assert!(root.join("scripts/Main.cs").is_file());

        let csproj = std::fs::read_to_string(root.join("scripts/TestGame.csproj")).expect("csproj");
        // A tempdir is outside any checkout, so the bindings must be referenced
        // by path rather than by a relative ProjectReference that would only
        // resolve on the machine that ran `orrin new`.
        assert!(csproj.contains("<HintPath>$(OrrinBindings)/Orrin.dll</HintPath>"));
        assert!(!csproj.contains("ProjectReference"));
        // Without this, `Debug.DrawLine` compiles away even in a Debug build.
        assert!(csproj.contains("ORRIN_DEBUG"));
        // A second copy of Orrin.dll in the game's output would mean a second
        // set of types, and every Behaviour cast across the boundary failing.
        assert!(csproj.contains(r#"Private="false""#));
        // ORRIN001 comes from the analyzer laid beside the bindings, and a
        // bindings directory without one must still build.
        assert!(
            csproj.contains(
                r#"<Analyzer Include="$(OrrinBindings)/Orrin.Analyzers.dll" Condition="Exists('$(OrrinBindings)/Orrin.Analyzers.dll')" />"#
            ),
            "{csproj}"
        );
    }

    /// The checkout shape, which `examples/hello-orrin` is: both the bindings
    /// and the analyzer by relative project path.
    #[test]
    fn a_project_inside_a_checkout_references_the_analyzer_project() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkout = dir.path();
        for marker in ["crates/core/Cargo.toml", "scripting/Orrin/Orrin.csproj"] {
            write(&checkout.join(marker), "").expect("marker");
        }
        std::fs::create_dir(checkout.join("examples")).expect("examples");
        scaffold("test-game", &checkout.join("examples")).expect("scaffold");

        let csproj =
            std::fs::read_to_string(checkout.join("examples/test-game/scripts/TestGame.csproj"))
                .expect("csproj");
        assert!(
            csproj.contains(
                r#"<ProjectReference Include="../../../scripting/Orrin/Orrin.csproj" Private="false" />"#
            ),
            "{csproj}"
        );
        // `ReferenceOutputAssembly="false"` is what keeps the analyzer, and the
        // Roslyn it is built on, out of the game assembly's references.
        assert!(
            csproj.contains(
                r#"<ProjectReference Include="../../../scripting/Orrin.Analyzers/Orrin.Analyzers.csproj"
                      OutputItemType="Analyzer" ReferenceOutputAssembly="false" />"#
            ),
            "{csproj}"
        );
        assert!(!csproj.contains("<Analyzer Include"), "{csproj}");
        assert!(!csproj.contains("HintPath"), "{csproj}");
    }

    /// The game assembly runs inside the engine's process, so a debugger is
    /// attached to that rather than launched.
    #[test]
    fn a_scaffolded_project_can_attach_a_debugger_to_the_engine() {
        let dir = tempfile::tempdir().expect("tempdir");
        scaffold("test-game", dir.path()).expect("scaffold");
        let root = dir.path().join("test-game");

        let text = std::fs::read_to_string(root.join(".vscode/launch.json")).expect("launch.json");
        let launch: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let attach = launch["configurations"]
            .as_array()
            .expect("a configurations array")
            .iter()
            .find(|c| c["type"] == "coreclr" && c["request"] == "attach")
            .unwrap_or_else(|| panic!("no coreclr attach configuration: {text}"));
        assert_eq!(attach["processName"], "orrin-core", "{text}");

        let readme = std::fs::read_to_string(root.join("README.md")).expect("README");
        assert!(readme.contains("## Debugging"), "{readme}");
        assert!(readme.contains("Rider"), "{readme}");
    }

    /// `Orrin.Native` is the bindings' own plumbing, not API a game may call.
    #[test]
    fn the_scaffolded_scripts_use_only_the_public_api() {
        let dir = tempfile::tempdir().expect("tempdir");
        scaffold("test-game", dir.path()).expect("scaffold");
        let scripts = dir.path().join("test-game/scripts");

        for file in ["Main.cs", "FlyCamera.cs"] {
            let source = std::fs::read_to_string(scripts.join(file)).expect(file);
            assert!(!source.contains("Native."), "{file}:\n{source}");
        }
    }

    /// A new project can be walked around as soon as it runs, in the editor's
    /// Play and in an export, where the editor's own fly camera is gone.
    #[test]
    fn a_scaffolded_project_attaches_a_wasd_camera() {
        let dir = tempfile::tempdir().expect("tempdir");
        scaffold("test-game", dir.path()).expect("scaffold");
        let scripts = dir.path().join("test-game/scripts");

        let camera = std::fs::read_to_string(scripts.join("FlyCamera.cs")).expect("FlyCamera.cs");
        assert!(camera.contains("namespace TestGame;"), "{camera}");
        assert!(camera.contains("class FlyCamera : Behaviour"), "{camera}");
        for axis in ["\"Horizontal\"", "\"Vertical\"", "\"LookX\"", "\"LookY\""] {
            assert!(camera.contains(axis), "reads {axis} from input.toml");
            assert!(INPUT.contains(&format!("[axes.{}]", axis.trim_matches('"'))));
        }
        let main = std::fs::read_to_string(scripts.join("Main.cs")).expect("Main.cs");
        assert!(main.contains("World.AddScript<FlyCamera>"), "{main}");
    }

    #[test]
    fn an_existing_non_empty_directory_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let occupied = dir.path().join("taken");
        std::fs::create_dir(&occupied).expect("create");
        std::fs::write(occupied.join("keep.txt"), "mine").expect("write");

        scaffold("taken", dir.path()).expect_err("must not clobber existing files");
        assert_eq!(
            std::fs::read_to_string(occupied.join("keep.txt")).unwrap(),
            "mine"
        );
    }
}
