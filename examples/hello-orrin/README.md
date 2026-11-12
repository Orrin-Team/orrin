# hello-orrin

A hand-written Orrin project showing the canonical on-disk layout. This is the
reference `orrin new` generates from; if the two drift, this directory is the
one that is right.

```
hello-orrin/
  orrin.toml            project manifest — the engine reads this from cwd (or any parent)
  assets/                source assets + .meta sidecars
    shaders/rising_bands.oshader   a surface shader
    shaders/embers.oshader         another, with helper functions
    shaders/water.oshader          a third, which writes the normal
    materials/*.omat               one material drawn through each
    scenes/surface_shaders.orrin   a scene of objects drawn with them
  scripts/               C# game code
    HelloOrrin.csproj   builds HelloOrrin.dll, this project's game assembly
    Main.cs              the entry Behaviour named by `scripts.entry`
    Tests.cs             a test `orrin test` runs, with no window
  scenes/                .fscene files
  .vscode/launch.json   attaches VS Code's C# debugger to the engine
  .orrin/               engine-generated, gitignored: cache + asset index
```

The project owns its game assembly. `HelloOrrin.dll` is loaded into a
collectible load context, separate from the engine's `Orrin.dll` bindings, so
it can be swapped while the engine runs.

## Running it

Through the CLI, which does all of the below and needs no environment setup:

```bash
cargo build -p orrin-cli
cd examples/hello-orrin && ../../target/debug/orrin run
```

By hand:

```bash
cargo build -p orrin-core
dotnet build scripting/Orrin
dotnet build examples/hello-orrin/scripts
```

```bash
cd examples/hello-orrin && ORRIN_SCRIPT_DIR=../../scripting/Orrin/bin/Debug/net10.0 ../../target/debug/orrin-core
```

The engine prints the project it picked up:

```
orrin: project `hello-orrin` at /…/examples/hello-orrin
```

Nothing points the engine at the game assembly: it derives the DLL's name from
the assembly half of `scripts.entry` (`HelloOrrin.Main, HelloOrrin`) and finds
it under `scripts/bin/`.

Built `--no-default-features` the manifest is still located and validated, so
that is also the quickest way to check manifest errors on a machine with no
.NET.

## Hot reload

With the engine running, edit `scripts/Main.cs` — change `SpinSpeed`, say —
then:

```bash
dotnet build examples/hello-orrin/scripts
```

and press **Reload scripts** in the editor's Scripts panel. The new code takes
effect without restarting, and `_spin` carries across the swap so the cube
picks up where it left off rather than snapping back to zero.

A build that fails, or leaves a DLL the runtime rejects, is refused: the reload
is staged before anything is torn down, so the session keeps running the code it
already had and says so in the console.

## Writing a surface shader

Double-click `scenes/surface_shaders.orrin` in the Scenes tab, or in the Assets
tab, to open a scene built to show them. A sphere and a block share
`rising_bands.omat`, and the bands run through both at one height because the
shader measures in world space. Beside them a sphere of coals is drawn through
`embers.oshader`, which adds no light of its own and only decides where its
material's emission is. The pool is a flat plane whose waves are
`water.oshader` tipping its normal: `water.omat` is Transmissive, so the paved
floor and the ball seen through it bend with every ripple. All three move in the
editor without pressing Play, and saving a shader changes the scene.

Press Play and `Main` also spawns a sphere beside the cube, drawn with
`assets/materials/rising_bands.omat`, which names the surface shader
`assets/shaders/rising_bands.oshader`: bands of light climbing the object in
world space over a dark, rough base. A surface
shader says what a material's surface *is*; the engine still lights, shadows and
reflects it like any other material.

A `.oshader` is GLSL holding `param` lines, one `surface` function and any
helpers it calls:

```glsl
param color glow = (1.0, 0.35, 0.08, 1.0);
param float speed = 0.5 : range(0, 4);

void surface(in SurfaceIn i, inout SurfaceOut o) {
    float line = smoothstep(0.7, 1.0, sin(i.world_position.y * 20.0 - i.time * speed));
    o.emissive += glow.rgb * 600.0 * line;
}
```

**Parameters** are what a material sets in the inspector, one per line,
outside any `{}`: `param <type> <name> [= <default>] [: <hint>, ...];`

| Type | Default written as | Omitted, it is | Hints |
| --- | --- | --- | --- |
| `float` | `0.5` | `0` | `range(min, max)`: the inspector's slider |
| `vec2` `vec3` `vec4` | `(x, y, ...)` | zeros | |
| `color` | `(r, g, b)` or `(r, g, b, a)`, linear | opaque white | |
| `texture` | `white`, `black` or `normal` | `white` | `srgb` or `linear`: how a dropped image imports |

A parameter is a global of its name inside the function. Read a texture with
`sample_param(name, uv)`.

**`SurfaceIn i`** is where the surface is: `uv` (after any parallax),
`world_position`, `world_normal` (the geometric normal, unit length),
`view_dir` (towards the eye, unit length), `vertex_color`, and `time` in
seconds.

**`SurfaceOut o`** arrives holding what the material alone would draw, so
change only what the shader is for: `albedo`, `alpha`, `metallic`, `roughness`
(perceptual), `emissive` (cd/m², the unit of every light in the engine — a
lit wall is about a hundred) and `normal` (tangent space). The engine clamps
what comes back to the ranges a material can have.

**Not allowed**, each an error at its line:

- writing `gl_FragDepth`: depth is settled before the surface is shaded;
- `discard`: lower `o.alpha` on a material whose blend is Masked instead, so
  every pass cuts along the same line;
- a `layout`, `uniform` or `buffer`: parameters are the shader's only inputs;
- `#version`, `#extension`, `#include` or `#line`: the engine's pass shader owns
  them;
- a parameter named like something the engine's shaders already declare.

A Masked material casts the shadow of its standard alpha, not of the alpha the
shader writes.

**Editing.** Double-click the shader in the Assets tab, or press Edit on the
material inspector's Shader row, to open it in the Shader tab: highlighted
code, its errors at their lines, the parameters it declares and a sphere drawn
with it. Save (or Cmd/Ctrl+S) compiles it and every material using it changes
within a frame or two. Any other editor works as well: the engine watches the
file and recompiles it on save. **New surface shader** in the Assets tab writes
a commented starting point.

**Errors** are listed in the Shader tab and logged as `path:line: message`. A
shader that does not compile changes nothing on screen: the last version that
compiled keeps drawing, and a material whose shader has never compiled draws as
a standard material until it does.

A material stores only the values that differ from the shader's defaults
(`rising_bands.omat` sets `glow` and `speed`, the other two nothing), so changing a default in the
shader reaches every material that did not override it. A value for a
parameter the shader no longer declares is kept in the file and listed in the
inspector to be removed.

## Testing

```bash
cd examples/hello-orrin && ../../target/debug/orrin test
```

builds the scripts and runs `scripts/Tests.cs` inside the engine with no
window: `Main` attached to an empty world, a frame for its OnStart to spawn the
cube, then a second's worth of frames for the cube to turn `SpinSpeed` degrees.
The command exits non-zero if a test fails. A test's world has no asset
database, so `Main`'s lookup of the sphere's material logs a warning there and
the sphere stays clay.

## Debugging

`HelloOrrin.dll` runs inside the engine's process, so a debugger attaches to the
running engine rather than launching the game. Start it with `orrin run` (or by
hand, as above) on a Debug build of the scripts — a Release build optimises
locals away — then:

- **VS Code**, with the C# extension, opened on this directory: Run and Debug,
  **Attach to Orrin**. It finds the engine by its process name, `orrin-core`;
  **Pick a process to attach to** asks instead.
- **Rider**: Run ▸ Attach to Process…, and pick `orrin-core`.

The engine loads `HelloOrrin.pdb` together with the DLL, which is what lets a
breakpoint in `scripts/` bind to the code that is running.

## Warnings about Behaviour fields

The build reports ORRIN001 for a Behaviour field the component registry cannot
save and that is not marked `[Transient]`: such a field silently fails to
survive a save, and may not survive a hot reload. The analyzer behind it is a
project reference here; a project outside this repository finds
`Orrin.Analyzers.dll` beside the engine's `Orrin.dll`, where `orrin build` puts
it, and builds without the warning if it is missing.

## Why `ORRIN_SCRIPT_DIR` is still needed

Only when launching the engine by hand — `orrin run` resolves the bindings and
passes them itself. Only for the *engine's* bindings, not for the game. The engine looks for
`Orrin.dll` beside its own executable — the layout an exported build ships —
and otherwise probes `scripting/Orrin/bin/` relative to the working directory,
which doesn't resolve from inside a project. The override goes away with export
packaging (issue #29). Game assembly discovery, project discovery from any
subdirectory, manifest validation, and the entry Behaviour all work without it.

## Path rules

Every path in `orrin.toml` is relative to this directory. Absolute paths and
`..` are rejected at load — nothing committed to version control may contain
machine-local paths.
