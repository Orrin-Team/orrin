# AGENTS.md

Working guide for Orrin. It applies to humans and to AI agents equally; where a
rule exists only because an agent is doing the work, it says so.

Read [`docs/architecture.md`](docs/architecture.md) before making a design
decision. It is the ten-year plan, and most "why is it like this?" questions are
answered there. This file is the day-to-day: layout, commands, and the standards
a change is held to.

---

## 1. What Orrin is

A Rust game engine: Vulkan renderer (vulkano 0.35), a small hand-written ECS,
an egui editor, and C# game code hosted in-process through CoreCLR.

Rust edition 2024, resolver 3. Cargo workspace, `members = ["crates/*"]`.

| Crate | What it owns |
| --- | --- |
| `crates/core` (`orrin-core`) | The engine and editor binary: renderer, render graph, scene, systems, editor panels, scripting FFI |
| `crates/orrin-ecs` | Entities, sparse-set storage, queries. `#![forbid(unsafe_code)]`. No renderer/math/window deps — mechanism only |
| `crates/orrin-registry` | Per-component reflection: read, write, remove, diff, apply, inspect, default, keyed by a stable string id. Scene save/load, the inspector, and later undo and sync are all written once against it. A C# Behaviour registers here too, through a property bag (`wire`) rather than a Rust type |
| `crates/orrin-macros` | `#[derive(Reflect)]` for the registry |
| `crates/orrin-script` | Boots CoreCLR via netcorehost; the C ABI over the engine |
| `crates/orrin-build` | `dotnet build` for a project's C# |
| `crates/orrin-project` | The `orrin.toml` project manifest |
| `crates/orrin-cli` (binary `orrin`) | `orrin new` / `build` / `run` |
| `scripting/Orrin` | The C# bindings assembly. `scripting/DemoGame` is a game assembly like any project's; `scripting/Orrin.Analyzers` is the Roslyn analyzer holding Behaviour fields to the registry-visible / `[Transient]` split |

Inside `crates/core/src`:

- `app.rs` — winit event loop, window, per-frame driver
- `systems.rs` — world sweep producing `FrameGeometry` (culling, cascades, lights)
- `gfx/graph/` — the render graph: passes declare `(resource, access)`, the
  compiler derives order, layouts and barriers. Takes no `Device`, so CI with no
  GPU asserts the exact barrier sequence
- `gfx/vulkan/` — one file per pass (forward, prepass, ssao, ssr, taa, bloom,
  dof, fog, oit, shadow, …) plus `frame.rs`, `record.rs`, `rendering.rs`
- `gfx/headless.rs` — `HeadlessBackend`, the CPU-only path CI tests through
- `scene/` — components, resources, model import, persistence, entity builders
- `editor/` — egui panels, dock, theme
- `profile.rs` / `stats.rs` — frame profiler (CPU phases + GPU passes) and HUD
- `logging.rs` — the one log stream: `tracing` events from the engine and
  `Debug.Log*` from scripts, out to stderr, the editor console and an export's
  log file. Log with `tracing::{info, warn, error}!`, never `println!`
- `threads.rs` — the engine's *one* rayon pool. Read its module docs before
  parallelising anything

---

## 2. Commands

```bash
# Build and test everything
cargo build --workspace
cargo test  --workspace

# The engine on its built-in demo (build the C# once first)
dotnet build scripting/Orrin && dotnet build scripting/DemoGame
cargo run -p orrin-core

# The no-.NET build shape. CI checks it; so should you if you touch a
# `#[cfg(feature = "scripting")]` seam.
cargo check -p orrin-core --no-default-features --all-targets

# A project through the CLI
cargo build -p orrin-cli && ./target/debug/orrin new my-game && ./target/debug/orrin run

# The C# half (math conventions, blittable struct layout, and the property-bag
# wire format — no Rust test sees any of it)
dotnet run --project scripting/Orrin.MathTests
```

Measurement:

```bash
# Frame cost, per CPU phase and per GPU pass. Needs a GPU, hence #[ignore].
cargo test --release -p orrin-core --test perf -- --ignored --nocapture

# Microbenchmarks
cargo bench -p orrin-core --bench bvh      # also: collision, startup
cargo bench -p orrin-ecs  --bench ecs

# The cold-start gate architecture §6 asks for
cargo test -p orrin-core --test cold_start -- --nocapture
```

Useful environment variables (full list in each harness's module docs):
`ORRIN_SCENE`, `ORRIN_STRESS`, `ORRIN_THREADS`, `ORRIN_VALIDATION`, `ORRIN_LOG`,
`ORRIN_PERF_*`, `ORRIN_UPDATE_GOLDEN`, `ORRIN_COLD_START_BUDGET_MS`.

CI (`.github/workflows/rust.yml`) runs: workspace build, workspace test, the
`--no-default-features` check, both C# builds, the C# math/ABI tests, and the
cold-start benchmark. A change is not done until all of those would pass.

---

## 3. Performance is a correctness property

This is a game engine. "Works" and "is fast" are the same requirement, and a
new feature ships in its fastest reasonable shape rather than in a shape that
gets optimised later — later never comes, and by then the content authored
against it makes the change expensive.

**Data is small and laid out for the loop that reads it.**

- Pick the narrowest type that is correct. Indices are `u32` unless something
  demands more.
- Split a struct along how it is *read*, not how it is authored. `gfx::Vertex`
  is one authoring struct that uploads as `PositionVertex` + `SurfaceVertex`,
  because depth-only passes would otherwise pull 60 bytes to use 12.
- Sort and index rather than copy. `FrameGeometry` keeps `Vec<u32>` orderings
  into one `items` array; materialising each list would copy 144-byte items five
  times a frame.
- Prefer `Vec3A` over `Vec3` where the value is SIMD-operated on in bulk;
  prefer `Vec3` where it is stored by the thousand.
- No allocation in the frame loop. Reuse buffers across frames.

**Before you optimise, measure. After you optimise, measure again.**

- `tests/perf.rs` prints the table an optimisation is chosen against. Choose
  against the table, never against an intuition about which pass is expensive.
- A threaded change is measured by running the harness twice *in one session*,
  `ORRIN_THREADS=1` against the pool — not against a number from last week.
- Report the numbers in the PR/commit body. A perf claim without a before/after
  is not a perf claim.
- An optimisation that measures as zero gets reverted, not kept "because it
  can't hurt". It costs reading time forever.

**GPU-side rules.** Passes declare their resource access and let the graph
derive barriers — never hand-write a barrier in a pass. New passes are graph
nodes; `PassKind::Raw` is an escape hatch with exactly one legitimate user
(egui). Keep vendor-specific work (`gfx/vulkan/vendor.rs`) behind a measured
justification.

**Unsafe.** `orrin-ecs` forbids it. Elsewhere it is confined to the FFI surface
(`scripting.rs`, `orrin-script`) and to raw ash recording (`record.rs`). Every
`unsafe` block carries a comment stating the invariant that makes it sound. If
you want unsafe somewhere new, that is a design discussion, not a diff.

---

## 4. Tests

**Agents write the test first.** Before implementing anything — a feature, a
fix, a refactor with observable behaviour — write the failing test, run it, see
it fail for the reason you expect, then implement until it passes. A patch that
arrives with its tests written afterwards has only proven the code agrees with
itself. This rule is for AI agents specifically; it exists because an agent's
implementation is persuasive prose and the test is the only thing that isn't.

Where tests live:

- **Unit tests inline**, in a `#[cfg(test)] mod tests` at the bottom of the file
  they cover. This is the default and where most of the ~400 tests are.
- **`crates/core/tests/`** for cross-module behaviour: `render_graph.rs` (barrier
  sequences, no GPU needed), `cold_start.rs` (the §6 startup gate),
  `culling.rs`, `hierarchy_probe.rs`, `offscreen.rs` and `perf.rs` (both
  `#[ignore]`d — they need a GPU).
- **Golden files** in `crates/core/tests/golden/`, refreshed with
  `ORRIN_UPDATE_GOLDEN=1`. Read the diff before you accept a new golden.
- **`benches/`** with criterion, for anything on the frame's critical path.
- **`proptest`** in `orrin-ecs` for handle and storage invariants.

What a good test asserts: the invariant, not the implementation. The render
graph tests assert the *barrier sequence*, which is the thing that would be
wrong. Prefer one test that would catch a real bug over five that restate the
code.

If a change genuinely cannot be tested (a Vulkan path with no headless
equivalent, an editor layout), say so explicitly in the PR and describe how you
verified it by hand instead. Don't quietly skip.

---

## 5. Comments

The codebase's comments are its documentation, and they are written to a
specific standard. Match it.

**A comment explains *why*, or explains something ambiguous.** The code already
says what it does.

```rust
// Good — states the reason the reader could not have derived:
/// The pool is one thread short of `available_parallelism`, because the thread
/// that calls into it is not idle while it waits — rayon runs part of the split
/// on the caller.

// Bad — restates the line below it:
// Increment the counter
counter += 1;
```

Rules of thumb:

- **Module docs (`//!`) carry the design.** Why this module exists, what shape
  it committed to, and what it deliberately does not do. This is where the
  long-form explanation belongs — see `gfx/graph/mod.rs`, `threads.rs`,
  `profile.rs`.
- **Item docs (`///`) on public items**, one or two sentences, plus the
  non-obvious constraint if there is one.
- **Inline comments are rare** and reserved for a line that would otherwise read
  as a mistake.
- **Document the rejected alternative** when it is the thing a future reader
  would try. "Not X, because Y" prevents the same wrong turn twice.
- **Length is bounded by usefulness.** Comments must not take over the file. If
  a paragraph is explaining code, consider whether clearer code removes the need
  for the paragraph. Commit `87f183a` was a cleanup of exactly this failure mode
  after an agent over-commented; do not recreate it.
- **Never narrate the change.** No "changed from X", "new implementation",
  "TODO(agent)", "as requested". Comments describe the code as it is, for
  someone reading it in two years with no memory of the diff.
- Prose is British-inflected and plain (`colour`, `rasteriser`). Full sentences.

---

## 6. Conventions

- **Plain data, addressed by handles.** No component holds a pointer or a direct
  reference to another entity or asset — only `Entity`/`EntityId`/`AssetId`.
  This one rule is what keeps scenes serializable, diffable and syncable
  (architecture §2.1). A violation is a correctness bug, not a style nit.
- **Adding a component**: define it, `#[derive(Reflect)]`, register it in
  `scene/registry.rs::register_components` under a stable id. Registration is
  explicit and re-runnable — linker-based auto-registration does not survive the
  dynamic library boundary hot reload creates. A C# component is the same rule
  from the other side: `[Component("game.thing")]` on the Behaviour, and its id
  is likewise never derived from the type's name.
- **Editing a component** goes through `diff` and `apply`, never a bare `write`.
  Undo, prefab overrides and (later) collaboration sync are all consumers of that
  one change stream (architecture §4.4), and an edit that skips it is invisible
  to all three.
- **Adding a field to an existing component**: `#[reflect(default)]`, or every
  scene saved before today fails to load. Removing or repurposing a field is a
  breaking change and is taken loudly (see the `intensity` → physical-units
  break in architecture §3.3), never silently reinterpreted.
- **Adding a render pass**: a graph node in `gfx/graph`, a module in
  `gfx/vulkan/`, declared resource access, and a `tests/render_graph.rs` case.
- **Errors name the thing.** Every user-facing failure names the entity, asset
  or pass involved and suggests a likely fix (architecture §1, §6).
- **Formatting**: `cargo fmt` (default rustfmt). `cargo clippy` clean;
  `orrin-ecs` additionally warns on `missing_docs` and all of clippy.
- **Commits**: `type: summary` in lower case — `feat:`, `fix:`, `perf:`,
  `ref:`, `chore:`, `style:`, optionally scoped (`style(editor):`). Reference
  the issue where there is one (`(#72)`).
- **Invariants that never regress** (architecture §6): C# hot reload survives
  every later system; shader edits swap within a frame or two; asset edits
  propagate without a restart; editor cold start stays under a few seconds.

---

## 7. Review

**Every AI-authored change is reviewed by the human who requested it before it
lands.** No exceptions, and the agent's job includes making that review possible.

An agent finishing a change presents, in the final message:

1. **What changed**, file by file, in one line each.
2. **Why**, where the reason is not obvious from the diff.
3. **The tests** — which were written first, what they assert, and the actual
   output of running them (pasted, not summarised as "tests pass").
4. **The numbers**, if anything on the frame path moved: before and after, from
   `tests/perf.rs` or a bench, with the environment stated.
5. **What was not done**: anything skipped, assumed, or left untested, and why.
6. **The riskiest part of the diff** — where the agent would look first if
   something broke. Point the reviewer at it by `file:line`.

Do not commit, push, or open a PR unless the human asks. Do not mark work
complete on the strength of a clean compile. "It builds" is not a result.

---

## 8. Scope

Do the change that was asked for. Not the tidy-up next to it, not the
refactor it suggests, not the extra feature it would obviously enable. If you
find something else worth doing, finish the task and then say so in one line.

Unasked-for scope in an AI patch is the most expensive thing to review, because
the reviewer has to work out which parts were requested.
