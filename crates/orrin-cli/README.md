# `orrin` — project CLI

Create, build, and run Orrin projects without knowing the engine's internals.
The crate is `orrin-cli`; the binary it installs is **`orrin`**.

```bash
cargo build -p orrin-cli
```

## Commands

### `orrin new <name> [--path DIR]`

Scaffolds a project directory, matching the canonical layout in
`examples/hello-orrin`:

```
<name>/
  orrin.toml            format_version 1, with scripts.entry pre-wired
  .gitignore            bin/, obj/, .orrin/
  README.md
  assets/               .gitkeep
  scenes/               .gitkeep
  scripts/
    <Name>.csproj       AssemblyName matches the entry type's assembly half
    Main.cs             the entry Behaviour — a spinning cube
```

The name is turned into a PascalCase assembly name (`space-shooter` →
`SpaceShooter`), used for the namespace, `<AssemblyName>`, the `.csproj`
filename, and the assembly half of `scripts.entry`. Names that cannot produce a
valid C# identifier are rejected rather than silently mangled. An existing
non-empty directory is never overwritten.

The generated `.csproj` references the engine bindings one of two ways, decided
by where the *project* lives — never by where the CLI binary lives, since the
reference gets committed:

- **Inside an engine checkout** — a relative `ProjectReference` to
  `scripting/Orrin/Orrin.csproj`, like `examples/hello-orrin`.
- **Anywhere else** — `<Reference Include="Orrin">` with its `HintPath` under
  `$(OrrinBindings)`, which `orrin build` supplies and `$ORRIN_SCRIPT_DIR`
  falls back to for a bare `dotnet build`.

Both use `Private="false"`, so no second `Orrin.dll` lands in the game's output.

### `orrin build [--release] [--project DIR]`

1. Locates the project by walking up from the current directory for
   `orrin.toml`.
2. Builds the engine bindings if this is a checkout that has never built them —
   a game assembly cannot compile without `Orrin.dll`.
3. Runs `dotnet build` on `scripts.dir`, passing the resolved bindings path.
4. Writes `.orrin/assets.toml`: every file under `assets.dir` with its size,
   mtime, and whether a `.meta` sidecar sits beside it.

Step 4 is an **index, not a package** — nothing is copied or transformed. It
exists so the command contract and the output location are settled before the
asset pipeline lands; the importer replaces the step without changing the
interface. Sidecars are recorded as a flag on the asset they describe, dotfiles
and symlinks are skipped, and entries are sorted so the file is identical
whichever machine produced it.

### `orrin run [--release] [--no-build] [--project DIR] [-- ARGS…]`

`orrin build`, then launch the engine with its working directory set to the
project root — which is how the engine finds the project, since it walks up
from cwd for `orrin.toml`. `--no-build` launches what is already compiled.
Arguments after `--` are forwarded to the engine. The engine's exit status
becomes the CLI's.

## How the engine is found

In order, because a wrong guess here is near-invisible:

1. **`$ORRIN_ENGINE`** — an explicit path. A value pointing at a non-file is an
   error, never a silent fallback.
2. **A shipped install** — `orrin-core` and `Orrin.dll` sitting *together*
   beside the CLI. That pairing is what an exported build lays down.
3. **An engine checkout** above the project or above the CLI itself →
   `cargo run -p orrin-core --features scripting`.
4. **`orrin-core` alone** beside the CLI.

Step 2 requires the bindings specifically so that a cargo `target/` directory —
where the engine and CLI binaries are neighbours but no `Orrin.dll` is — falls
through to step 3. Otherwise a contributor's `orrin run` would launch whatever
stale `target/debug/orrin-core` existed, possibly one built without
`--features scripting`, and scripts would simply never run with nothing said.

## `$ORRIN_SCRIPT_DIR`

`orrin run` resolves the bindings directory and passes it to the engine, unless
the variable is already set. This is what makes running from inside a project
directory work in a checkout: the engine's own fallback probes
`scripting/Orrin/bin/` relative to cwd, which does not resolve once cwd is the
project. Setting it by hand — the workaround `examples/hello-orrin/README.md`
documents — is no longer needed when launching through the CLI.

## Not yet

- `orrin new` creates `scenes/` but no scene file: the format does not exist
  yet (issue #6). Generating a `.fscene` the engine cannot load would be worse
  than an empty directory, so the manifest's `[scenes]` section is left
  commented out.
- Export/packaging is Phase 5; `build` deliberately stops at the index.
