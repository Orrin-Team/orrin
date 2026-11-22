# CLAUDE.md

**[`AGENTS.md`](AGENTS.md) is the source of truth** for how this codebase is
built, tested, commented and reviewed. Read it. This file is the operating
procedure on top of it: how a Claude Code session in this repo should run.

Design questions are answered by the `//!` header of the module concerned. When
a change and that header disagree, the header wins or it gets updated in the
same change — never neither.

Use judgement. Most of this file is facts that are expensive to get wrong, not
a procedure to walk through. How a change is approached is yours to decide;
what follows is what it is held to.

---

## What a change is held to

1. **It is the better option, not the first one that works.** See the next
   section.
2. **Measured performance.** This is a game engine. New code lands in its
   fastest reasonable shape, data stays small, and any perf claim comes with
   before/after numbers from `tests/perf.rs` or a bench. (AGENTS.md §3)
3. **Comments explain why, and stay short.** Module docs carry the design;
   inline comments are rare. Never narrate the diff. Commit `87f183a` was a
   cleanup after exactly this went wrong — don't cause the sequel. (AGENTS.md §5)
4. **The human reviews everything.** Finish with the review report in
   AGENTS.md §7 and stop. No commits, no pushes, no PRs unless asked.
5. **It stays on the change requested.** A fix the change cannot work without
   is part of it. An adjacent tidy-up gets mentioned, not made.

---

## Is this the better option?

Ask it before implementing a feature, and of a rendering feature above all. A
technique that works is not the bar; the best one Orrin can afford is.

- Compare it with what it replaces or competes with: what Orrin does today, and
  what shipped games, engines and papers do for the same problem.
- Judge it by what the player sees and pays for: image quality, stability in
  motion, frame cost, memory, and how it sits in a clustered forward renderer.
- If a better option exists, say so before writing code, with the reason and
  what it costs. Then build the one the human picks.
- If the request already is the best option, one line saying why is enough.
  Don't manufacture alternatives to look thorough.

---

## Tests: few, and only where they pay

Tests are not written first by rule, and most changes need one or none.
(AGENTS.md §4)

- Write one where a bug would lose work, corrupt data, or break an invariant CI
  can check: save and load, undo, the registry, the render graph's barrier
  plan, ECS storage, the C# ABI.
- Don't write tests for how something looks, for editor layout or wording, or
  ones that restate the code. A rendering change is judged from captures, not
  from a pixel assertion.
- One test that would catch a real bug, not one per phrasing of it.

The existing suite still has to pass.

---

## Rendering changes come with captures

Any change that moves pixels is delivered as before and after images the human
loads into NVIDIA ICAT. They are the evidence for the change, in place of tests.

```bash
cargo test -p orrin-core --test offscreen -- --ignored --nocapture
```

1. Before editing, run it and copy `target/capture/` to `target/icat/before/`.
2. After the change, run it again and copy to `target/icat/after/`.
3. If no built-in capture shows the change, add one to
   `crates/core/tests/common/mod.rs` that does: the right scene, the feature
   on, the camera where the difference is. Where a setting toggles the feature,
   capture it on and off from the same build instead.

Keep the pair honest: the same file name, resolution, camera and warm-up on
both sides, and nothing differing but the change. In the report, give both
folder paths, the files to open first, and what to look for in them.

A still cannot show ghosting, shimmer or lag. When the change is temporal, say
that the captures do not cover it and how to see it in the running editor.

---

## How a session should go

**Before writing code.** Read the module you are about to change, including its
`//!` header — the reason for its shape is usually written there and usually not
guessable.

**While working.** Prefer `cargo check -p <crate>` for the fast loop and reserve
`cargo build --workspace` for the end.

**Before reporting done**, run what CI runs:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo check -p orrin-core --no-default-features --all-targets   # if you touched a scripting cfg seam
cargo check -p orrin-core --no-default-features --features scripting,export --all-targets   # if you touched a shader-compiler cfg seam
```

If you added a test, paste its real output. For a rendering change, give the
capture folders. (AGENTS.md §7, item 3)

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
- **Surface shaders.** A `.oshader` changes what a surface *is*, never how it
  is lit, and it is one function every surface-reading pass runs — so
  `SurfaceIn` holds nothing pass-specific and the normal leaves in tangent
  space. A custom draw binds exactly what its built-in twin binds plus one
  parameter buffer that changes only when a value does; a binding of a
  shader's own, or a per-frame value in that buffer (the time is not there, on
  purpose), is a design change. An artifact read from disk is trusted only by
  its integrity and sources hashes, and a test never hands a driver SPIR-V that
  frames and parses and is still invalid: MoltenVK crashes the process on it.
  `gfx/surface/mod.rs` is the design.
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
- Don't touch `assets/sponza/` (fetched on demand), `.orrin/`, or `target/`
  beyond the capture folders above.
- Don't claim completion on a clean compile. Compiling is not a result.
