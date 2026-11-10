# hello-orrin

A hand-written Orrin project showing the canonical on-disk layout. This is the
reference `orrin new` generates from; if the two drift, this directory is the
one that is right.

```
hello-orrin/
  orrin.toml            project manifest — the engine reads this from cwd (or any parent)
  assets/                source assets + .meta sidecars
  scripts/               C# game code
    HelloOrrin.csproj   builds HelloOrrin.dll, this project's game assembly
    Main.cs              the entry Behaviour named by `scripts.entry`
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
