# CLAUDE.md

**[`AGENTS.md`](AGENTS.md) is the source of truth** for how this codebase is
built, tested, commented and reviewed. Read it. This file is the operating
procedure on top of it: how a Claude Code session in this repo should run.

Design questions are answered by the `//!` header of the module concerned. When
a change and that header disagree, the header wins or it gets updated in the
same change — never neither.

---

## The five non-negotiables

1. **Tests first.** Write the failing test, run it, watch it fail for the right
   reason, then implement. Not afterwards. (AGENTS.md §4)
2. **Measured performance.** This is a game engine. New code lands in its
   fastest reasonable shape, data stays small, and any perf claim comes with
   before/after numbers from `tests/perf.rs` or a bench. (AGENTS.md §3)
3. **Comments explain why, and stay short.** Module docs carry the design;
   inline comments are rare. Never narrate the diff. Commit `87f183a` was a
   cleanup after exactly this went wrong — don't cause the sequel. (AGENTS.md §5)
4. **The human reviews everything.** Finish with the review report in
   AGENTS.md §7 and stop. No commits, no pushes, no PRs unless asked.
5. **Only the change requested.** Adjacent tidy-ups get mentioned, not made.

---

## How a session should go

**Before writing code.** Read the module you are about to change, including its
`//!` header — the reason for its shape is usually written there and usually not
guessable. Check whether a test harness already covers the area
(`crates/core/tests/`, the inline `mod tests`, `benches/`).

**While working.** Prefer `cargo check -p <crate>` for the fast loop and reserve
`cargo build --workspace` for the end. Run the tests you wrote as you go, not in
one batch at the finish.

**Before reporting done**, run what CI runs:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo check -p orrin-core --no-default-features --all-targets   # if you touched a scripting cfg seam
```

Paste the real output of the tests you added. "Tests pass" summarised is not
evidence; the harness's own lines are.

**If you touched the frame path**, also run the perf harness and quote the
table:

```bash
ORRIN_STRESS=4000 cargo test --release -p orrin-core --test perf -- --ignored --nocapture
```

Compare against a run you did in the *same session*. Numbers from a previous
session are not a baseline.

---

## Things that will be wrong if you guess

- **Barriers.** Passes declare `(resource, access)`; the graph compiler derives
  order, layouts and barriers. Never hand-write one in a pass. Add a
  `tests/render_graph.rs` case for any new node.
- **Threading.** There is exactly one rayon pool, built in `threads.rs`, sized
  one short of the machine on purpose, and `ORRIN_THREADS=1` is a genuinely
  serial path rather than a pool of one. Read that module before adding
  parallelism. Profiler scopes are suppressed on pool workers.
- **Component fields.** A new field on an existing component needs
  `#[reflect(default)]` or every previously saved scene fails to load. A removed
  or repurposed field is a loud break, never a silent reinterpretation.
- **Entity references.** Components hold `Entity`/`EntityId`/`AssetId`, never
  pointers or direct references. This is what makes scenes diffable and
  syncable; breaking it breaks three future features at once.
- **Light and colour units.** Lights are physical (lux, lumens, cd/m²) end to
  end, and exposure is EV100 over real luminance. A "just scale it by 0.001"
  fix is always wrong here.
- **The C# side.** `sizeof(Transform) == 40` and the math conventions
  (right-handed, -Z forward, degrees, YXZ euler, Hamilton products) are a frozen
  ABI checked only by `dotnet run --project scripting/Orrin.MathTests`. Touching
  a blittable struct means running it.
- **`unsafe`.** Confined to FFI (`scripting.rs`, `orrin-script`) and raw ash
  recording (`record.rs`); forbidden in `orrin-ecs`. New `unsafe` anywhere else
  is a conversation before it is a diff.

---

## Don't

- Don't delete or rewrite an existing comment block to make room for your own —
  those paragraphs are the design record.
- Don't add `TODO`, `FIXME`, or any comment addressed to a future agent.
- Don't "fix" a failing test by changing its assertion. Work out which side is
  wrong and say so.
- Don't accept a regenerated golden file (`ORRIN_UPDATE_GOLDEN=1`) without
  reading the diff and explaining it.
- Don't add a dependency without saying what it costs — cold start is a CI-gated
  budget and it decays one dependency at a time.
- Don't touch `assets/sponza/` (fetched on demand), `target/`, or `.orrin/`.
- Don't claim completion on a clean compile. Compiling is not a result.
