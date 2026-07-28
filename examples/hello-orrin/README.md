# hello-orrin

A hand-written Orrin project showing the canonical on-disk layout. There is no
`orrin new` yet (issue #17), so this directory doubles as the reference for
what that command will eventually generate.

```
hello-orrin/
  orrin.toml            project manifest — the engine reads this from cwd (or any parent)
  assets/                source assets + .meta sidecars
  scripts/               C# game code
    HelloOrrin.csproj   builds HelloOrrin.dll, this project's game assembly
    Main.cs              the entry Behaviour named by `scripts.entry`
  scenes/                .fscene files
  .orrin/               engine-generated, gitignored: cache + asset index
```

The project owns its game assembly. `HelloOrrin.dll` is loaded into a
collectible load context, separate from the engine's `Orrin.dll` bindings, so
it can be swapped while the engine runs.

## Running it

```bash
cargo build -p orrin-core --features scripting
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

Without `--features scripting` the manifest is still located and validated, so
this is also the quickest way to check manifest errors.

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

## Why `ORRIN_SCRIPT_DIR` is still needed

Only for the *engine's* bindings, not for the game. The engine looks for
`Orrin.dll` beside its own executable — the layout an exported build ships —
and otherwise probes `scripting/Orrin/bin/` relative to the working directory,
which doesn't resolve from inside a project. The override goes away with export
packaging (issue #29). Game assembly discovery, project discovery from any
subdirectory, manifest validation, and the entry Behaviour all work without it.

## Path rules

Every path in `orrin.toml` is relative to this directory. Absolute paths and
`..` are rejected at load — nothing committed to version control may contain
machine-local paths.
