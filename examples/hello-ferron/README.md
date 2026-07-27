# hello-ferron

A hand-written Ferron project showing the canonical on-disk layout. There is no
`ferron new` yet (issue #17), so this directory doubles as the reference for
what that command will eventually generate.

```
hello-ferron/
  ferron.toml     project manifest — the engine reads this from cwd (or any parent)
  assets/         source assets + .meta sidecars
  scripts/        C# game code (.csproj will live here)
  scenes/         .fscene files
  .ferron/        engine-generated, gitignored: cache + asset index
```

## Running it

```bash
cargo build -p renderer-prototype --features scripting
dotnet build scripting/Ferron
cd examples/hello-ferron
FERRON_SCRIPT_DIR=../../scripting/Ferron/bin/Debug/net10.0 ../../target/debug/renderer-prototype
```

The engine prints the project it picked up:

```
ferron: project `hello-ferron` at /…/examples/hello-ferron
```

Without `--features scripting` the manifest is still located and validated, so
this is also the quickest way to check manifest errors.

## The bootstrap caveat

`FERRON_SCRIPT_DIR` is needed above because a project cannot yet own its game
assembly — the runtime still loads types out of the engine's own `Ferron`
bindings assembly (`Ferron.Demo.Game, Ferron` is a type inside it), and
`scripts/` here is empty. That is exactly what issue #5 (hot reload + external
game assemblies) changes: once a project builds its own `game.dll`, assembly
discovery follows `scripts.dir` and the override goes away.

What already works without the override: project discovery from any
subdirectory, manifest validation, and the entry Behaviour coming from
`scripts.entry` rather than `FERRON_ENTRY`.

## Path rules

Every path in `ferron.toml` is relative to this directory. Absolute paths and
`..` are rejected at load — nothing committed to version control may contain
machine-local paths.
