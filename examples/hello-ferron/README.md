# hello-ferron

A hand-written Ferron project showing the canonical on-disk layout. There is no
`ferron new` yet (issue #17), so this directory doubles as the reference for
what that command will eventually generate.

```
hello-ferron/
  ferron.toml            project manifest — the engine reads this from cwd (or any parent)
  assets/                source assets + .meta sidecars
  scripts/               C# game code
    HelloFerron.csproj   builds HelloFerron.dll, this project's game assembly
    Main.cs              the entry Behaviour named by `scripts.entry`
  scenes/                .fscene files
  .ferron/               engine-generated, gitignored: cache + asset index
```

The project owns its game assembly. `HelloFerron.dll` is loaded into a
collectible load context, separate from the engine's `Ferron.dll` bindings, so
it can be swapped while the engine runs.

## Running it

```bash
cargo build -p renderer-prototype --features scripting
dotnet build scripting/Ferron
dotnet build examples/hello-ferron/scripts
```

```bash
cd examples/hello-ferron && FERRON_SCRIPT_DIR=../../scripting/Ferron/bin/Debug/net10.0 ../../target/debug/renderer-prototype
```

The engine prints the project it picked up:

```
ferron: project `hello-ferron` at /…/examples/hello-ferron
```

Nothing points the engine at the game assembly: it derives the DLL's name from
the assembly half of `scripts.entry` (`HelloFerron.Main, HelloFerron`) and finds
it under `scripts/bin/`.

Without `--features scripting` the manifest is still located and validated, so
this is also the quickest way to check manifest errors.

## Hot reload

With the engine running, edit `scripts/Main.cs` — change `SpinSpeed`, say —
then:

```bash
dotnet build examples/hello-ferron/scripts
```

and press **Reload scripts** in the editor's Scripts panel. The new code takes
effect without restarting, and `_spin` carries across the swap so the cube
picks up where it left off rather than snapping back to zero.

A build that fails, or leaves a DLL the runtime rejects, is refused: the reload
is staged before anything is torn down, so the session keeps running the code it
already had and says so in the console.

## Why `FERRON_SCRIPT_DIR` is still needed

Only for the *engine's* bindings, not for the game. The engine looks for
`Ferron.dll` beside its own executable — the layout an exported build ships —
and otherwise probes `scripting/Ferron/bin/` relative to the working directory,
which doesn't resolve from inside a project. The override goes away with export
packaging (issue #29). Game assembly discovery, project discovery from any
subdirectory, manifest validation, and the entry Behaviour all work without it.

## Path rules

Every path in `ferron.toml` is relative to this directory. Absolute paths and
`..` are rejected at load — nothing committed to version control may contain
machine-local paths.
