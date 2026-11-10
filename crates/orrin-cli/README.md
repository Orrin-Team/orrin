# `orrin` — project CLI

Create, build, run and export Orrin projects without knowing the engine's
internals.
The crate is `orrin-cli`; the binary it installs is **`orrin`**.

```bash
cargo build -p orrin-cli
```

## Putting `orrin` on `$PATH`

**While developing the engine**, symlink the dev build — the link keeps pointing
at whatever `cargo build -p orrin-cli` last produced, so there is no reinstall
step:

```bash
ln -sf "$PWD/target/debug/orrin" ~/.local/bin/orrin
```

Symlinks are resolved before the checkout is probed, so this behaves the same
on macOS and Linux even though `current_exe` reports the link on one and the
target on the other.

**`cargo install --path crates/orrin-cli`** also works, but it *copies* the
binary to `~/.cargo/bin`, severing it from the checkout. Projects inside the
checkout still work (the project's own location finds it), but a project
anywhere else has no engine to run and needs `$ORRIN_ENGINE` set. That is the
right trade only once there are real engine releases to install against.

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
  .vscode/launch.json   attach a C# debugger to the running engine
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

The Behaviour-field analyzer (ORRIN001) is referenced the same two ways: as an
analyzer `ProjectReference` to `scripting/Orrin.Analyzers` in a checkout, and
as `$(OrrinBindings)/Orrin.Analyzers.dll` elsewhere, conditional on the file
existing so bindings without it still build. Neither is ever copied into the
game's output.

### `orrin build [--release] [--project DIR]`

1. Locates the project by walking up from the current directory for
   `orrin.toml`.
2. Builds the engine bindings if this is a checkout that has never built them —
   a game assembly cannot compile without `Orrin.dll`. When the bindings are a
   checkout's build output, the analyzer built in the same configuration is
   copied beside them; it is not part of the bindings' publish, so an export
   never carries it.
3. Runs `dotnet build` on `scripts.dir`, passing the resolved bindings path.
4. Writes `.orrin/assets.toml`: every file under `assets.dir` with its size,
   mtime, and whether a `.meta` sidecar sits beside it.

Step 4 is an **index, not a package** — nothing is copied or transformed. It
exists so the command contract and the output location are settled before the
asset pipeline lands; the importer replaces the step without changing the
interface. Sidecars are recorded as a flag on the asset they describe, dotfiles
and symlinks are skipped, and entries are sorted so the file is identical
whichever machine produced it.

### `orrin run [--release] [--dev] [--no-build] [--project DIR] [-- ARGS…]`

`orrin build`, then launch the engine with its working directory set to the
project root — which is how the engine finds the project, since it walks up
from cwd for `orrin.toml`. `--no-build` launches what is already compiled.

From an engine checkout the engine is built with `--release`, so the editor runs
at the speed the game will; the first launch compiles it, later ones reuse it.
`--dev` runs the dev build instead, which compiles faster and runs slower.
`--release` is about the *scripts*: it builds them in Release, which strips
`Debug.DrawLine` and friends, so leave it off while editing.
Arguments after `--` are forwarded to the engine. The engine's exit status
becomes the CLI's.

### `orrin export [--platform mac|linux|windows] [--out DIR] [--project DIR]`

Packages the project as a directory a player can run on a machine with no
.NET, no Rust and no Vulkan SDK. `orrin build --export` is the same command.

```
exports/<name>-<platform>/
  <name>[.exe]       the engine, release, `export` feature, symbols stripped
  Orrin.dll          the bindings, with their runtimeconfig
  <Game>.dll         the game assembly, built Release (no ORRIN_DEBUG)
  dotnet/            a private .NET runtime the engine boots instead of any install
  compiled-assets/   cache artifacts, sidecars and text assets; no sources
  orrin.json         which game this is: entry, assembly, start scene
```

1. **Publishes the C#** for the target's runtime identifier, and assembles a
   `DOTNET_ROOT` from the NuGet runtime pack that a self-contained publish
   restores. A Windows runtime can be built on a Mac this way.
2. **Builds the engine** for the target, from the checkout so its
   `rust-toolchain.toml` applies: `cargo` when it is this machine, `cargo xwin`
   for Windows elsewhere. A Mac game can only be built on a Mac. A Linux game
   from elsewhere needs `cargo zigbuild` *and* a Linux sysroot with libudev
   (the gamepad backend links it) named by `PKG_CONFIG_SYSROOT_DIR`; without
   one the export is refused up front and CI's `linux` package is the way.
   Missing tools and Rust targets are checked before anything slow starts; a
   missing Rust target is added for you.
3. **Packs the assets** with the checkout's own engine on this machine,
   importing anything the cache lacks. The pack copies artifacts byte for byte,
   so exporting twice from one import gives the same files.
4. **Assembles** the directory and writes `orrin.json`.

The engine is built from source, so exporting needs the `orrin` from an engine
checkout. `--out` defaults to `exports/<name>-<platform>` in the project, which
`orrin new` gitignores. An earlier export there is replaced; any other non-empty
directory is refused rather than cleared.

A scene named in `[scenes]` is found in the pack when it sits under the assets
directory, and is copied to the same relative path when it does not.

Mac exports are Apple Silicon only: that is where the Metal backend is the
default. An Intel Mac would need MoltenVK shipped beside the engine.

CI's `export` job runs this once per platform, each on its own OS, and uploads
the playable directory.

### `orrin hub`

Opens Orrin Hub (`crates/orrin-hub`), a window for creating, opening, building
and exporting projects. Every button runs one of the commands above, and the
hub shows that command's output, so it can never do something the CLI cannot.
The hub found beside `orrin` is used; otherwise the checkout `orrin` came from
builds and runs it. Recent projects are kept in `hub.json` in the platform's
per-user settings directory.

## How the engine is found

In order, because a wrong guess here is near-invisible:

1. **`$ORRIN_ENGINE`** — an explicit path. A value pointing at a non-file is an
   error, never a silent fallback.
2. **A shipped install** — `orrin-core` and `Orrin.dll` sitting *together*
   beside the CLI. That pairing is what an exported build lays down.
3. **An engine checkout** above the project or above the CLI itself (with the
   CLI's path canonicalized first, so a symlink on `$PATH` still points into
   the checkout) → `cargo run --release -p orrin-core` (without `--release`
   under `orrin run --dev`).
4. **`orrin-core` alone** beside the CLI.

Step 2 requires the bindings specifically so that a cargo `target/` directory —
where the engine and CLI binaries are neighbours but no `Orrin.dll` is — falls
through to step 3. Otherwise a contributor's `orrin run` would launch whatever
stale `target/debug/orrin-core` existed rather than a rebuild of the sources
they are editing.

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
- Exporting from an installed `orrin` with no engine checkout: there are no
  prebuilt per-platform engines to package yet.
